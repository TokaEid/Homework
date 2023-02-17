"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from scipy.integrate._ivp import common



class ForwardEuler(scipy.integrate.OdeSolver): 
    """
    Calculating ForwardEuler steps to solve ODE
    """
    
    def __init__(self, fun, t0, y0, t_bound, vectorized, h=None, support_complex=False, **extraneous):
        
        # Extraneous variables warning
        common.warn_extraneous(extraneous)
        
        # Calling on OdeSolver methods
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=False)
        
        # if h isn't given by user, set it to default, else set it to the user's value
        if h is None:
            self.h = (t_bound-t0)/100
        else:
            self.h = h
            
        # Array of all ts and ys    
        self.ts = [t0]
        self.ys = [y0]
            

    def _step_impl(self):
        
        # Saving old values of t and y
        t_old = self.t
        y_old = self.y
        
        # Moving one step forward by calculating the new y 
        y_new = y_old + self.h * self.fun(t_old, y_old)
        
        # Iterating t and saving the new y value
        self.t = t_old + self.h
        self.y = y_new
        
        # Appending calculated t and y in the ts and ys arrays
        self.ts.append(self.t)
        self.ys.append(self.y[0])
        
        return True, None
    
    
    def _dense_output_impl(self):
                
        return ForwardEulerOutput(self.ts, self.ys)
        


class ForwardEulerOutput(DenseOutput):
    """
    Interpolate ForwardEuler output

    """
    def __init__(self, ts, ys):

        """
        store ts and ys computed in forward Euler method

        These will be used for evaluation
        """
        super(ForwardEulerOutput, self).__init__(np.min(ts), np.max(ts))
        self.interp = interp1d(ts, ys, kind='linear', copy=True)


    def _call_impl(self, t):
        """
        Evaluate on a range of values
        """
        return self.interp(t)

    