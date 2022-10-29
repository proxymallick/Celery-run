import json
import numpy as np

from celery import Celery

#from celery import current_app
#from celery.contrib.methods import task_method
#CELERY_ACCEPT_CONTENT = ['pickle']
app = Celery('tasks', broker='pyamqp://guest@localhost//')
#app = Celery('vms', broker='redis://localhost', include=['cve.tasks','cpe.tasks'])
#app = Celery('tasks', broker='redis://localhost:6379/0',result_backend = "redis://localhost/0")
#####
## Author: PM 23/06/2021
## Generic Biobjective optimiser but can be easily adapted to multiobjective by 
## declaring and adding objective functions and different parameters 
#####

import math
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import minimize
import sys
import pdb
import matplotlib
import matplotlib.pyplot as plt
import plotly.figure_factory as ff  # type: ignore
seed_generator = 1
#import tensorflow as tf
#import autograd.numpy as np
#from autograd import grad, jacobian, hessian

#tf.enable_eager_execution()

values = []
param=[]
method = 'SLSQP'
bounds = [[0.5326,2.2532]] # [[-1.e2,1.e2]] # 
gamma_step = 1/(3*(np.sqrt(200)))
font = {'family' : 'normal',
'weight' : 'bold',
'size'   : 22}
objectives = 2
a = 1  # constant that does not influence optimised result (1 in article)
f = [0, 0]  # normalised objective function value
w = [0, 0]  # normalised weight function
n =(objectives)+3
beta = 1.1
restrictivePref  =False
init_guess = np.random.uniform( bounds[0][0],bounds[0][1] ) #1
storeP = []
P = None
args = None

@app.task()
def runAggregateObjFunc(serialisable):
    args = json.loads(serialisable)
    gamma1 = args['gamma1']
    gamma_step = args['gamma_step']
    bounds = args['bounds']
    min_obj =  (  args['min_obj'] )
    max_obj =  ( args['max_obj'] )
    #portfolios = args['portfolios']
    #port_shocks = args['port_shocks']
    S_f = args['S_f']
    S_p = args['S_p']
    E = args['E']
    P_0 = args['P_0']
    
    #individual_returns = args['individual_returns']
    frontier = []
    for gamma2 in np.arange(0, 1 - gamma1 + gamma_step, gamma_step):
        #gamma3 = 1-gamma1-gamma2
        
        # Create gamma array
        gamma = np.array([[gamma1, 0 ], [0, gamma2 ] ])
        gamma_cmp = np.array([[1-gamma1, 0 ], [0, 1-gamma2 ] ])
        #print (gamma.shape)
        #print (gamma_cmp.shape)
        #print (min_obj[0])

        min_obj = np.array(min_obj[0]).reshape ( (1,2))
        max_obj = np.array(max_obj[0]).reshape ( (1,2))

        # Create Translation Vector
        S_t = np.dot(min_obj, gamma) + np.dot(max_obj,gamma_cmp)
        S_tf = S_f + S_t.T + S_p
        #print (P_0.shape , E.shape,S_f.shape,S_t.shape,S_p.shape,S_tf.shape)
        P = np.dot(S_tf, E) + P_0
        args['P'] = P
        global seed_generator
        want_random_init = True
        np.random.seed(seed_generator)
        seed_generator += 1
        #if want_random_init:
        #    init_guess = np.array(np.random.dirichlet(np.ones( len(portfolios)), size=1))
        #init_guess =  np.array( [0.053,0.154,0.525,0.051,0.052,0.163])
        #var1 = tf.Variable(0.0)
        # Create an optimizer with the desired parameters.
        #opt = tf.keras.optimizers.SGD(learning_rate=0.1)
        #for i in range(100):
        #    opt_op = opt.minimize(self.normalisedWeightFunction , var_list=[var1])
        #start = tf.constant([0.9])  # Starting point for the search.
        #optim_results = tfp.optimizer.bfgs_minimize(
        #    quadratic_loss_and_gradient, initial_position=start, tolerance=1e-8)

        #pdb.set_trace()
        GaAlgo  =0

        #jacobian_  = jacobian(self.normalisedWeightFunction)
        if not GaAlgo:
            try:
                #i=1
                #while i < 1000:
                #    result = minimize( self.normalisedWeightFunction,self.init_guess,method=self.method\
                #        ,bounds=self.bounds,    options = {'maxiter': 1000,'disp':False})
                #    i=i*1.2
                #p = Pool(4)
                #pdb.set_trace()
                result =  minimize(  normalisedWeightFunction, init_guess,args = (args,),method= method,bounds=bounds,   options = {'maxiter': 1000,'disp':False})  
            except ValueError as e:
                result = {"success": False, "message": str(e)}
        else:
            try:
                pass
                #result = differential_evolution( self.normalisedWeightFunction, self.bounds, seed = 1, maxiter=1000, \
                #    disp=False, args=args)
            except ValueError:
                result = {"success": False}

        param.append(result['x'])  
    #print(param)        
    return result                    


def normalisedWeightFunction(weights ,args ):

    weights = np.array(weights)
    i = 0

    o1 = get_ret_heuristic(weights) [1]    
    o2 =  get_ret_heuristic(weights) [2]    
    order = [o1, o2]
    num = 0
    denom = 0
    P=args['P']
    #pdb.set_trace()
    storeP.append(P)
    for i in range(len(P)):

        val = order[i]
        pref = P[i]
            
        upper = 0
        lower = 0
        #print (pref)
        #pdb.set_trace()
        lamb= np.random.uniform(0,1,1)
        if(val > pref[4]):
            return sys.maxsize
            # print("specific result: "+ str(abs(val - pref[4]) * 1000))
        elif(val < pref[0]):
            #f[i] = a* np.random.uniform(0,np.exp(  (val-pref[0])/(pref[1]-pref[0])) ,1)  
            f[i] = a* np.exp(  (val-pref[0])/(pref[1]-pref[0]))
        else:
            for j in range(len(pref)):
                preference = pref[j]
                if(preference > val):
                    break
            k = j + 1
            upper = pref[j]
            lower = pref[j-1]
            f[i] = (k-1)*a + a*(val-lower)/(upper-lower)

        # Calculate weight
        if(f[i] != 200):
            w[i] = (beta*(n-1))**(f[i]/2)
        else:
            w[i] = sys.maxsize

    for i in range(len(w)):
        num += w[i]*f[i]
        denom += w[i]
    outVal = num/denom
    values.append( outVal)  
    
    return outVal

  
def get_ret_heuristic( weights):
    weights = np.array(weights)
    obj1 = np.sin( weights )
    obj2 = 1 - np.sin(weights)**7
    #obj1 = weights**2
    #obj2 = (weights-2)**2
    return np.array([ 0 , obj1, obj2,0])
def minimiseObj2( weights):
    return  get_ret_heuristic(weights)[2]
def minimiseObj1(  weights):
    return  get_ret_heuristic(weights)[1]

def maximiseObj1( weights):
    return - get_ret_heuristic(weights)[1]

def maximiseObj2( weights):
    return -get_ret_heuristic(weights)[2]
def constraint1(x):
    return 1.2532-x
def constraint2( x):
    return x-0.5326
def getMinMaxCaseStudy():
    #where the inequalities are of the form C_j(x) >= 0.

    con1 = ({'type': 'ineq', 'fun':  constraint1},
            {'type': 'ineq', 'fun': constraint2 })
    columns = ['obj1', 'obj2']
    index = ['min', 'max']
    minMax = pd.DataFrame(index=index, columns=columns, dtype=object)

    minObj2 = minimize( minimiseObj2, init_guess, method=method,
                    bounds= bounds )['x']
    results = get_ret_heuristic(minObj2) [2]  
    
    minObj2 = results 

    maxObj2 = minimize(maximiseObj2, init_guess, method=method,
                    bounds=bounds )['x']
    results =  get_ret_heuristic(maxObj2) [2]  
    maxObj2 = results 

    minObj1 = minimize(minimiseObj1, init_guess,
                        method=method, bounds=bounds )['x']
    minObj1 = get_ret_heuristic(minObj1) [1]  

    maxObj1 = minimize(maximiseObj1, init_guess,
                        method=method, bounds=bounds )['x']
    maxObj1 = get_ret_heuristic(maxObj1) [1]   
    
    if restrictivePref:

        minMax['obj2']['min'] = 0.4 #minObj2
        minMax['obj2']['max'] = 0.8 #maxObj2
        minMax['obj1']['min'] =  0.5 # minObj1
        minMax['obj1']['max'] = 0.7 #maxObj1
    else: 
        minMax['obj2']['min'] = minObj2
        minMax['obj2']['max'] =  maxObj2
        minMax['obj1']['min'] =   minObj1
        minMax['obj1']['max'] = maxObj1
    
    return (minMax)


''' 
class paretoOptimiser():
    def __init__(self):
        self.values = []
        self.param=[]
        self.method = 'SLSQP'
        self.bounds = [[0.5326,2.2532]] # [[-1.e2,1.e2]] # 
        self.gamma_step = 1/(3*(np.sqrt(200)))
        self.font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
        self.objectives = 2
        self.a = 1  # constant that does not influence optimised result (1 in article)
        self.f = [0, 0]  # normalised objective function value
        self.w = [0, 0]  # normalised weight function
        self.n =(self.objectives)+3
        self.beta = 1.1
        self.restrictivePref  =False
        self.init_guess = np.random.uniform( self.bounds[0][0],self.bounds[0][1] ) #1
        self.storeP = []
        self.P = None
        self.args = None

    def setValues(self,val):
        self.values.append(val)
    def getValues(self):
        return self.values 
    def get_ret_heuristic(self,weights):
        weights = np.array(weights)
        obj1 = np.sin( weights )
        obj2 = 1 - np.sin(weights)**7
        #obj1 = weights**2
        #obj2 = (weights-2)**2
        return np.array([ 0 , obj1, obj2,0])
    def minimiseObj2(self,weights):
        return self.get_ret_heuristic(weights)[2]
    def minimiseObj1(self,weights):
        return self.get_ret_heuristic(weights)[1]

    def maximiseObj1(self,weights):
        return -self.get_ret_heuristic(weights)[1]

    def maximiseObj2(self,weights):
        return -self.get_ret_heuristic(weights)[2]
    def constraint1(self,x):
        return 1.2532-x
    def constraint2(self,x):
        return x-0.5326
    def getMinMaxCaseStudy(self):
        #where the inequalities are of the form C_j(x) >= 0.

        con1 = ({'type': 'ineq', 'fun': self.constraint1},
                {'type': 'ineq', 'fun': self.constraint2 })
        columns = ['obj1', 'obj2']
        index = ['min', 'max']
        minMax = pd.DataFrame(index=index, columns=columns)

        minObj2 = minimize(self.minimiseObj2, self.init_guess, method=self.method,
                        bounds=self.bounds )['x']
        results = self.get_ret_heuristic(minObj2) [2]  
        
        minObj2 = results 

        maxObj2 = minimize(self.maximiseObj2, self.init_guess, method=self.method,
                        bounds=self.bounds )['x']
        results =  self.get_ret_heuristic(maxObj2) [2]  
        maxObj2 = results 

        minObj1 = minimize(self.minimiseObj1, self.init_guess,
                            method=self.method, bounds=self.bounds )['x']
        minObj1 = self.get_ret_heuristic(minObj1) [1]  

        maxObj1 = minimize(self.maximiseObj1, self.init_guess,
                            method=self.method, bounds=self.bounds )['x']
        maxObj1 = self.get_ret_heuristic(maxObj1) [1]   
        
        if self.restrictivePref:
 
            minMax['obj2']['min'] = 0.4 #minObj2
            minMax['obj2']['max'] = 0.8 #maxObj2
            minMax['obj1']['min'] =  0.5 # minObj1
            minMax['obj1']['max'] = 0.7 #maxObj1
        else: 
            minMax['obj2']['min'] = minObj2
            minMax['obj2']['max'] =  maxObj2
            minMax['obj1']['min'] =   minObj1
            minMax['obj1']['max'] = maxObj1
        print (minMax)
        
        return minMax
    def normalisedWeightFunction(self,weights ):
 
        weights = np.array(weights)
        i = 0

        o1 = self.get_ret_heuristic(weights) [1]    
        o2 =  self.get_ret_heuristic(weights) [2]    
        order = [o1, o2]
        num = 0
        denom = 0
        P=self.P
        self.storeP.append(P)
        for i in range(len(P)):

            val = order[i]
            pref = P[i]
             
            upper = 0
            lower = 0
            #print (pref)
            #pdb.set_trace()
            lamb= np.random.uniform(0,1,1)
            if(val > pref[4]):
                return sys.maxsize
                # print("specific result: "+ str(abs(val - pref[4]) * 1000))
            elif(val < pref[0]):
                #self.f[i] = self.a* np.random.uniform(0,np.exp(  (val-pref[0])/(pref[1]-pref[0])) ,1)  
                self.f[i] = self.a* np.exp(  (val-pref[0])/(pref[1]-pref[0]))
            else:
                for j in range(len(pref)):
                    preference = pref[j]
                    if(preference > val):
                        break
                k = j + 1
                upper = pref[j]
                lower = pref[j-1]
                self.f[i] = (k-1)*self.a + self.a*(val-lower)/(upper-lower)

            # Calculate weight
            if(self.f[i] != 200):
                self.w[i] = (self.beta*(self.n-1))**(self.f[i]/2)
            else:
                self.w[i] = sys.maxsize

        for i in range(len(self.w)):
            num += self.w[i]*self.f[i]
            denom += self.w[i]
        outVal = num/denom
        self.values.append( outVal)  
        
        return outVal

    
    @app.task()
    def runAggregateObjFunc(self, serialisable):
        args = pickle.loads(serialisable)
        gamma1 = args['gamma1']
        gamma_step = args['gamma_step']
 
        min_obj = args['min_obj']
        max_obj = args['max_obj']
        #portfolios = args['portfolios']
        #port_shocks = args['port_shocks']
        S_f = args['S_f']
        S_p = args['S_p']
        E = args['E']
        P_0 = args['P_0']
        
        #individual_returns = args['individual_returns']
        frontier = []
        for gamma2 in np.arange(0, 1 - gamma1 + gamma_step, gamma_step):
            #gamma3 = 1-gamma1-gamma2
            
            # Create gamma array
            gamma = np.array([[gamma1, 0 ], [0, gamma2 ] ])
            gamma_cmp = np.array([[1-gamma1, 0 ], [0, 1-gamma2 ] ])
            #print (gamma.shape)
            #print (gamma_cmp.shape)
            min_obj = min_obj.reshape ( (1,2))
            max_obj = max_obj.reshape ( (1,2))

            # Create Translation Vector
            S_t = np.dot(min_obj, gamma) + np.dot(max_obj,gamma_cmp)
            S_tf = S_f + S_t.T + S_p
            #print (P_0.shape , E.shape,S_f.shape,S_t.shape,S_p.shape,S_tf.shape)
            P = np.dot(S_tf, E) + P_0
            self.P = P
            args = P
            global seed_generator
            want_random_init = True
            np.random.seed(seed_generator)
            seed_generator += 1
            #if want_random_init:
            #    init_guess = np.array(np.random.dirichlet(np.ones( len(portfolios)), size=1))
            #init_guess =  np.array( [0.053,0.154,0.525,0.051,0.052,0.163])
            #var1 = tf.Variable(0.0)
            # Create an optimizer with the desired parameters.
            #opt = tf.keras.optimizers.SGD(learning_rate=0.1)
            #for i in range(100):
            #    opt_op = opt.minimize(self.normalisedWeightFunction , var_list=[var1])
            #start = tf.constant([0.9])  # Starting point for the search.
            #optim_results = tfp.optimizer.bfgs_minimize(
            #    quadratic_loss_and_gradient, initial_position=start, tolerance=1e-8)

            #pdb.set_trace()
            GaAlgo  =0

            #jacobian_  = jacobian(self.normalisedWeightFunction)
            if not GaAlgo:
                try:
                    #i=1
                    #while i < 1000:
                    #    result = minimize( self.normalisedWeightFunction,self.init_guess,method=self.method\
                    #        ,bounds=self.bounds,    options = {'maxiter': 1000,'disp':False})
                    #    i=i*1.2
                    #p = Pool(4)
                    result =  minimize( self.normalisedWeightFunction,self.init_guess,method=self.method\
                        ,bounds=args['bounds'],   options = {'maxiter': 1000,'disp':False})  
                except ValueError as e:
                    result = {"success": False, "message": str(e)}
            else:
                try:
                    pass
                    #result = differential_evolution( self.normalisedWeightFunction, self.bounds, seed = 1, maxiter=1000, \
                    #    disp=False, args=args)
                except ValueError:
                    result = {"success": False}

            self.param.append(result['x'])          
        return result                    



    @app.task
    def runOpt():
        import numpy as np
        from scipy.optimize import minimize

        def objective(x):
            return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2] + x[2]**3 * x[1]

        def constraint1(x):
            return x[0]*x[1]*x[2]*x[3]-25.0

        def constraint2(x):
            sum_eq = 40.0
            for i in range(4):
                sum_eq = sum_eq - x[i]**2
            return sum_eq

        # initial guesses
        n = 4
        x0 = np.zeros(n)
        t_array = []
        
        x0[0] = 1.0 + np.random.rand()
        x0[1] = 5.0 + np.random.rand()
        x0[2] = 5.0+ np.random.rand()
        x0[3] = 1.0+ np.random.rand()

        # show initial objective
        #print('Initial SSE Objective: ' + str(objective(x0)))

        # optimize
        b = (1.0,5.0)
        bnds = (b, b, b, b)
        con1 = {'type': 'ineq', 'fun': constraint1} 
        con2 = {'type': 'eq', 'fun': constraint2}
        cons = ([con1,con2])

        solution = minimize(objective,x0,method='SLSQP',\
                            bounds=bnds,constraints=cons )
        #print (solution)
        #t_array += [solution.x]
        #x = solution.x

        # show final objective
        #print('Final SSE Objective: ' + str(objective(x)))

        # print solution
        #print('Solution')
        #print('x1 = ' + str(x[0]))
        #print('x2 = ' + str(x[1]))
        #print('x3 = ' + str(x[2]))
        #print('x4 = ' + str(x[3]))
        return solution.x
 
    @app.task
    def add(x, y):
        return x + y 
    '''