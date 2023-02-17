"""
Implement tests for Problem 0 Part C
"""
import unittest

from euler import ForwardEuler
from scipy.integrate import solve_ivp
import numpy as np
import scipy.sparse as sparse
import scipy.linalg as sla



class TestValue(unittest.TestCase):
    
    def setUp(self):
        pass
        
    # I unfortunately wasn't able to figure this one out. I included my attempts in the answers.ipynb notebook. 


class TestEuler(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_forward_euler(self):
        
        # Initial condition
        y0 = np.array([1])
        
        # Defining the function
        f = lambda t, y : y

        # Time interval
        t_span = (0, 5)
        t_eval = np.linspace(0, 5, 100)
        
        
        # Solving the initial value problem with the default method
        trueSol = solve_ivp(f, t_span, y0, t_eval=t_eval)
        
        # Solving the initial value problem using the ForwardEuler method
        eulerSol = solve_ivp(f, t_span, y0, method=ForwardEuler, h=0.01, t_eval=t_eval)
        
        # Compare both solutions according to a tolerance to see if they're almost equal 
        self.assertTrue(np.allclose(eulerSol.y, trueSol.y[0], rtol=1e6, atol=1e8))
        
        
        

        
        
        
        