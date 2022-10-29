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

    

    def runAggregateObjFunc(self,serialisable):
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
            args = self.P
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
                        ,bounds=self.bounds,   options = {'maxiter': 1000,'disp':False})  
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

    def plot(self,data,minMax ):
        matplotlib.rc('font', **self.font)
        colorscale = ['#7A4579', '#D56073',
                        'rgb(255, 237, 222)', (1, 1, 0.2), (0.8, 0.8, 0.98)]
        fig, ax =plt.subplots(1,1, figsize=(13,13),sharey=True)

        #ax.scatter(  np.sin(data), 1 - np.sin( data )**7 ,color = 'red')
        #ax.plot(  [minMax['obj1'][0],minMax['obj1'][1] ], [minMax['obj2'][1],minMax['obj2'][0] ],color = 'red')

        ax.set_xlabel( r'$f_1 =\sin \theta$' )
        ax.set_ylabel( r'$f_2 = 1 - \sin^7 \theta$' )
        ax.legend(['pareto solutions'],loc="upper right")
        #theta = np.linspace( 0, 2*math.pi,400)
        theta = np.linspace(-1.e3,1.e3,40000)
        fig, axGeneric =plt.subplots(1,1, figsize=(13,13),sharey=True)
        axGeneric.plot( theta,self.get_ret_heuristic(theta)[1],color = 'blue')
        axGeneric.plot( theta,self.get_ret_heuristic(theta)[2] ,color = 'cyan')
        axGeneric.set_xlim(self.bounds[0])
        axGeneric.set_ylim([0,100])
        #print (funArray)
        #print (fun.getValues())
        #pdb.set_trace()
        #sliceLen = 1000 #len(self.storeP)
        #slicedArr = self.storeP#[:sliceLen]
        #ax.plot( slicedArr[0],slicedArr[1] ,color = 'blue')
        #for i in range( sliceLen ):
        #    temp = slicedArr[i]
        #    ax.plot( temp[0] ,temp[1] )
        dat =  np.array(self.param) 
        #print (max(dat))
        ax.scatter( np.arange(len(dat)) , dat  )
        #print (min(dat))
        axGeneric.scatter(  self.get_ret_heuristic(dat)[1], self.get_ret_heuristic(dat)[2]  ,color = 'magenta')
        axGeneric.plot(  self.get_ret_heuristic(theta)[1] , self.get_ret_heuristic(theta)[2]  ,color = 'red')
        #pdb.set_trace()
        plt.show()



    def wppf(self, alpha,  minMax):

        min_obj = np.array(
            [[minMax['obj2']['min'], minMax['obj1']['min'] ] ])
        max_obj = np.array(
            [[minMax['obj2']['max'], minMax['obj1']['max'] ] ] )
        d = (max_obj-min_obj ).reshape( 2,1 )
        n_d = 3
        box = d/n_d
        print (box)
        
        frontier_weights = []
        frontier_data = pd.DataFrame()
        S_p = np.array([])
        # Create pseudopreference vector
        args = {}
        args['n_p'] = 0
        args['alpha'] = alpha
        args['min_obj'] = min_obj
        args['max_obj'] = max_obj
        
        t_array = []
        t_fun = []
        r_array = []
        for n_p in range(1, 6):

            args['n_p'] = n_p
            alpha_min = max([-0.5 + (n_p-1)/(4*n_d), -1])
            alpha_max = min([0.5 + (n_p-1)/(4*n_d), 1])
            #pdb.set_trace()
            for alpha_1 in [i for i in np.arange(alpha_min ,alpha_max , (alpha_max-alpha_min)/3 )]: #[0]:

                # Create offset vector
                S_f = alpha_1*d
                S_p = -(n_p-1)*0.25*box
                P_v = np.array([[0, 0.25, 0.5, 0.75, 1]])
                box = np.array(box)
                P_0 = np.dot(box, P_v)
                E = np.array([[1, 1, 1, 1, 1]])

                args['S_f'] = S_f
                args['S_p'] = S_p
                args['P_0'] = P_0
                args['E'] = E
                args['bounds'] = self.bounds
                args['gamma_step'] = self.gamma_step
                #pdb.set_trace()
                for gamma1 in np.arange(0, 1 + self.gamma_step, self.gamma_step):
                    args['gamma1'] = gamma1
                    serialisable = pickle.dumps(args)
                    #t = tasks.runAggregateObjFunc.delay(serialisable)
                    #t_array += [t]
                    t = self.runAggregateObjFunc(serialisable)
                    t_array += [t['x']]
                    t_fun += [t['fun']]

        self.plot (t_array,minMax)
        print(frontier_weights)

        return frontier_data

if __name__ == "__main__":
    p = paretoOptimiser()
    minMax = p.getMinMaxCaseStudy()

    p.wppf(0.5,minMax)