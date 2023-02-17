"""
A library of functions
"""
import numpy as np
import matplotlib.pyplot as plt
import numbers

class AbstractFunction:
    """
    An abstract function class
    """

    def derivative(self):
        """
        returns another function f' which is the derivative of x
        """
        raise NotImplementedError("derivative")


    def __str__(self):
        return "AbstractFunction"


    def __repr__(self):
        return "AbstractFunction"


    def evaluate(self, x):
        """
        evaluate at x

        assumes x is a numeric value, or numpy array of values
        """
        raise NotImplementedError("evaluate")


    def __call__(self, x):
        """
        if x is another AbstractFunction, return the composition of functions

        if x is a string return a string that uses x as the indeterminate

        otherwise, evaluate function at a point x using evaluate
        """
        if isinstance(x, AbstractFunction):
            return Compose(self, x)
        elif isinstance(x, str):
            return self.__str__().format(x)
        else:
            return self.evaluate(x)


    # the rest of these methods will be implemented when we write the appropriate functions
    def __add__(self, other):
        """
        returns a new function expressing the sum of two functions
        """
        return Sum(self, other)


    def __mul__(self, other):
        """
        returns a new function expressing the product of two functions
        """
        return Product(self, other)


    def __neg__(self):
        return Scale(-1)(self)


    def __truediv__(self, other):
        return self * other**-1


    def __pow__(self, n):
        return Power(n)(self)


    #Problem 0 Part A
    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        self.x = vals  
        #Values on the x axis, can be changed using the np.linspace in method input
        
        self.y = self.evaluate(vals)
        #Values on the y axis, calls on the evaluate method from that function's child class and calculates f(x) for each x
        
        plt.plot(self.x, self.y, **kwargs)
        
        return plt.show()
    
    #Problem 2 Part A
    def taylor_series(self, x0, deg=5):
        """
        Returns the Taylor series of f centered at x0 truncated to degree k.
        """
        f=self
        F=Constant(self(x0))
        for i in range(1,deg+1):
            g=f.derivative() #take the derivative
            f=g #update f
            
            c=np.math.factorial(i)
            pc=Constant(g(x0)/c) #coefficient for each term
            ppower=Compose(Power(i),Polynomial(1,-x0)) #(x-x0)^{i}
            F=Sum(F,Product(pc,ppower))
        
        return F


    
class Polynomial(AbstractFunction):
    """
    polynomial c_n x^n + ... + c_1 x + c_0
    """

    def __init__(self, *args):
        """
        Polynomial(c_n ... c_0)

        Creates a polynomial
        c_n x^n + c_{n-1} x^{n-1} + ... + c_0
        """
        self.coeff = np.array(list(args))


    def __repr__(self):
        return "Polynomial{}".format(tuple(self.coeff))


    def __str__(self):
        """
        We'll create a string starting with leading term first

        there are a lot of branch conditions to make everything look pretty
        """
        s = ""
        deg = self.degree()
        for i, c in enumerate(self.coeff):
            if i < deg-1:
                if c == 0:
                    # don't print term at all
                    continue
                elif c == 1:
                    # supress coefficient
                    s = s + "({{0}})^{} + ".format(deg - i)
                else:
                    # print coefficient
                    s = s + "{}({{0}})^{} + ".format(c, deg - i)
            elif i == deg-1:
                # linear term
                if c == 0:
                    continue
                elif c == 1:
                    # suppress coefficient
                    s = s + "{0} + "
                else:
                    s = s + "{}({{0}}) + ".format(c)
            else:
                if c == 0 and len(s) > 0:
                    continue
                else:
                    # constant term
                    s = s + "{}".format(c)

        # handle possible trailing +
        if s[-3:] == " + ":
            s = s[:-3]

        return s


    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = 0
            for k, c in enumerate(reversed(self.coeff)):
                ret = ret + c * x**k
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            # use vandermonde matrix
            return np.vander(x, len(self.coeff)).dot(self.coeff)


    def derivative(self):
        if len(self.coeff) == 1:
            return Polynomial(0)
        return Polynomial(*(self.coeff[:-1] * np.array([n+1 for n in reversed(range(self.degree()))])))


    def degree(self):
        return len(self.coeff) - 1


    def __add__(self, other):
        """
        Polynomials are closed under addition - implement special rule
        """
        if isinstance(other, Polynomial):
            # add
            if self.degree() > other.degree():
                coeff = self.coeff
                coeff[-(other.degree() + 1):] += other.coeff
                return Polynomial(*coeff)
            else:
                coeff = other.coeff
                coeff[-(self.degree() + 1):] += self.coeff
                return Polynomial(*coeff)

        else:
            # do default add
            return super().__add__(other)


    def __mul__(self, other):
        """
        Polynomials are clused under multiplication - implement special rule
        """
        if isinstance(other, Polynomial):
            return Polynomial(*np.polymul(self.coeff, other.coeff))
        else:
            return super().__mul__(other)
        
        
class Affine(Polynomial):
    """
    affine function a * x + b
    """
    def __init__(self, a, b):
        super().__init__(a, b)
        

#Problem 0 Part B
class Scale(Polynomial):
    """
    scale function a * x + 0
    """
    def __init__(self, a):
        b = 0
        #Defining b=0 ensures that we can have two inputs when calling the Polynomial class where the constant is always 0
        super().__init__(a, b)
        
        
        
class Constant(Polynomial):
    """
    constant function c
    """
    def __init__(self, c):
        #This one was simple since it only takes one argument and when that's given to Polynomial, it outputs exactly the 
        #type of function we want
        super().__init__(c)
        

#Problem 0 Part C
class Compose(AbstractFunction):
    """
    compose(f,g)(x) such that it's f(g(x))
    """
    def __init__(self,f,g):
        if isinstance(f, numbers.Number):
            f=Constant(f)
        if isinstance(g, numbers.Number):
            g=Constant(g)
        self.f=f
        self.g=g

    def __str__(self):
        return self.f.__str__().replace("{0}",self.g.__str__())
    
    def __repr__(self):
        return f"Compose({self.f},{self.g})"
    
    def derivative(self):
        """
        Returns the derivative of f(g(x)) as f'(g(x))*g'(x)
        """
        return Compose(self.f.derivative(),self.g)*self.g.derivative()
    
    def evaluate(self,x):
        return self.f(self.g(x))
    
    
    
class Product(AbstractFunction):
    """
    Product(f,g)(x) such that f(x)*g(x)
    """
    def __init__(self,f,g):
        if isinstance(f, numbers.Number):
            f=Constant(f)
        if isinstance(g, numbers.Number):
            g=Constant(g)
        self.f=f
        self.g=g

    def __str__(self):
        return "{} * {}".format(self.f, self.g)
    
    def __repr__(self):
        return f"Product({self.f},{self.g})"
    
    def derivative(self):
        """
        Returns the product rule where d/dx(f*g) = f'(x)g(x) + f(x)g'(x)
        """
        return Product(self.f.derivative(),self.g) + Product(self.f,self.g.derivative())
    
    def evaluate(self,x):
        return self.f(x)*self.g(x)
    
    
    
class Sum(AbstractFunction):
    """
    Sum(f,g)(x) such that f(x)+g(x)
    """
    def __init__(self,f,g):
        if isinstance(f, numbers.Number):
            f=Constant(f)
        if isinstance(g, numbers.Number):
            g=Constant(g)
        self.f=f
        self.g=g

    def __str__(self):
        return "{} + {}".format(self.f, self.g)
    
    def __repr__(self):
        return f"Sum({self.f},{self.g})"
    
    def derivative(self):
        """
        Returns the sum of the derivatives f'(x) + g'(x)
        """
        return Sum(self.f.derivative(),self.g.derivative())
    
    def evaluate(self,x):
        return self.f(x)+self.g(x)



#Problem 0 Part D
class Power(AbstractFunction):
    """
    power function c*x^n 
    """
    
    def __init__(self, n, a=1):
        """
        Power(n,a=1)

        Creates a power function
        a*x^n 
        """
        self.pwr = n
        self.coeff = a
        
    def __repr__(self):
        return "Power{}".format([self.pwr])
    
    def __str__(self):
        """
        Creates a string to show the coefficient of the expression and the power of an indeterminate {0}
        """
        return "{}*({{0}})^{}".format(self.coeff, self.pwr)
    
    def derivative(self):
        """
        Gets an expression for the derivative of a power function
        """
        if self.pwr == 0:
            # x^0 = 1 so it's derivative would just be 0
            return 0
        
        elif self.pwr == 1:
            # x^1 is the simplest form of this function and is derivative is always 1
            return self.coeff
        
        else:
            # In all other cases, the derivative would be another power function with the exponent being n-1
            # and the new coeff being the old one * n
            return Power(self.pwr-1, self.pwr*self.coeff)
        
    def evaluate(self, x):   
        """
        Evaluate power function at x
        """
        
        if isinstance(x, numbers.Number):
            # If x is a number we return the simple expression a * x^n
            return self.coeff * x**self.pwr
        
        elif isinstance(x, np.ndarray):
            # If x is an array, we use np.power to raise all elements of the array to the nth power and multiply by
            # the coefficient
            x = np.array(x)
            return np.power(x, self.pwr) * self.coeff
        
        
class Log(AbstractFunction):
    """
    Log function c0*log[c1*x] 
    """
    
    def __init__(self, a=1, b=1):
        """
        Log(a=1,b=1)

        Creates a log function
        b*log(a*x)
        """
        self.x_coeff = a
        self.l_coeff = b
        
    def __repr__(self):
        return "Log( )"
    
    def __str__(self):
        """
        Creates a string to show the coefficient of the expression and the log of an indeterminate {0} with its coefficient
        """
        return "{}*log[{}*{{0}}]".format(self.l_coeff,self.x_coeff)
    
    def derivative(self):
        """
        Gets an expression for the derivative of a log function
        """
        #Derivative of log(x) is 1/x so it's the Power function with n set to -1
        return Power(-1, self.x_coeff*self.l_coeff)
        
    def evaluate(self, x):   
        """
        Evaluate log function at x
        """
        
        if isinstance(x, numbers.Number):
            # If x is a number we calculate b*log(ax)
            return np.log(self.x_coeff * x) * self.l_coeff
        
        elif isinstance(x, np.ndarray):
            # If x is an array we take the log of each element of the array multiplied by a, then multiply the results by b
            x = np.array(x)
            return np.log(self.x_coeff * x) * self.l_coeff
        
        
class Exponential(AbstractFunction):
    """
    exponential function c0*exp[c1*x] 
    """
    
    def __init__(self, a=1, b=1):
        """
        Exponential(a=1,b=1)

        Creates an exponential function
        b*exp[a*x]
        """
        self.x_coeff = a
        self.e_coeff = b
        
    def __repr__(self):
        return "Exp( )"
    
    def __str__(self):
        """
        Creates a string to show the coefficient of the expression and the exponential of an indeterminate {0} with its coefficient
        """
        return "{}*exp[{}*{{0}}]".format(self.e_coeff, self.x_coeff)
    
    def derivative(self):
        """
        Gets an expression for the derivative of an exponential function
        """
        
        #Derivative of exp[ax] is a*exp[ax] so the x_coeff stays the same while the exponential coeff (e_coeff)
        #gets multiplied by the x_coeff
        return  Exponential(self.x_coeff, self.x_coeff*self.e_coeff)
        
    def evaluate(self, x):
        """
        Evaluate exponential function at x
        """
        
        if isinstance(x, numbers.Number):
            # If x is a number we calculate b*exp[ax]
            return self.e_coeff * np.exp(self.x_coeff * x)
        
        elif isinstance(x, np.ndarray):
            # If x is an array we calculate the exponential of each element of the array multiplied by a, then multiply the results by b
            x = np.array(x)
            return self.e_coeff * np.exp(self.x_coeff * x)


class Sin(AbstractFunction):
    """
    sine function c0*sin[c1*x] 
    """
    
    def __init__(self, a=1, b=1):
        """
        Sin(a=1,b=1)

        Creates a sine function
        b*sin[a*x]
        """
        self.x_coeff = a
        self.s_coeff = b
        
    def __repr__(self):
        return "Sin( )"
    
    def __str__(self):
        """
        Creates a string to show the coefficient of the expression and the sine of an indeterminate {0} with its coefficient
        """
        return "{}*sin[{}*{{0}}]".format(self.s_coeff, self.x_coeff)
    
    def derivative(self):
        """
        Gets an expression for the derivative of a sine function
        """
        
        #Derivative of sin(ax) is a*cos(ax) so we call on the Cos function with the same x_coeff and change the coefficient
        #of the expression to x_coeff*s_coeff
        return  Cos(self.x_coeff, self.x_coeff*self.s_coeff)
        
    def evaluate(self, x):   
        """
        Evaluate sine function at x
        """
        
        if isinstance(x, numbers.Number):
            # If x is a number we calculate b*sin[ax]
            return self.s_coeff * np.sin(self.x_coeff * x)
        
        elif isinstance(x, np.ndarray):
            # If x is an array we calculate the sin of each element of the array multiplied by a, then multiply the results by b
            x = np.array(x)
            return self.s_coeff * np.sin(self.x_coeff * x)

        
class Cos(AbstractFunction):
    """
    cosine function c0*cos[c1*x] 
    """
    
    def __init__(self, a=1, b=1):
        """
        Cos(a=1,b=1)

        Creates a cosine function
        b*cos[a*x]
        """
        self.x_coeff = a
        self.c_coeff = b
        
    def __repr__(self):
        return "Cos( )"
    
    def __str__(self):
        """
        Creates a string to show the coefficient of the expression and the cosine of an indeterminate {0} with its coefficient
        """
        return "{}*cos[{}*{{0}}]".format(self.c_coeff, self.x_coeff)
    
    def derivative(self):
        """
        Gets an expression for the derivative of a cosine function
        """
        
        #Derivative of cos(ax) is -a*sin(ax) so we call on the Sin function with the same x_coeff and change the coefficient
        #of the expression to - x_coeff*c_coeff
        return  Sin(self.x_coeff, -1*self.x_coeff*self.c_coeff)
        
    def evaluate(self, x):   
        """
        Evaluate cosine function at x
        """
        
        if isinstance(x, numbers.Number):
            # If x is a number we calculate b*cos[ax]
            return self.c_coeff * np.cos(self.x_coeff * x)
        
        elif isinstance(x, np.ndarray):
            # If x is an array we calculate the cos of each element of the array multiplied by a, then multiply the results by b
            x = np.array(x)
            return self.c_coeff * np.cos(self.x_coeff * x)

        
#Problem 0 Part E        
class Symbolic(AbstractFunction):
    """
    Symbolic function 'f'
    """
    def __init__(self, f):
        """
        Symbolic('f') gives a string 'f()'
        """
        self.funct = f
        
    def __str__(self):
        """
        Returns a string with the symbolic form of the function "f({0})"
        """
        return self.funct + "({0})"
    
    def evaluate(self, x):
        """
        Returns a string with the function being evaluated at a specific input "f(x)" 
        """
        return self.funct + "({})".format(x)
    
    def derivative(self):
        """
        Calls the Symbolic class to output the prime of the function "f'({0})"
        """
        return Symbolic(self.funct + "'")

#Problem 1 Part A
def newton_root(f, x0, tol=1e-8):
    """
    find a point x so that f(x) is close to 0,
    measured by abs(f(x)) < tol

    Use Newton's method starting at point x0
    """
    x_n = x0
    for n in range(0,10000):
        if abs(f.evaluate(x_n)) < tol:
            return x_n
        Dfxn = f.derivative().evaluate(x_n)
        if Dfxn == 0:
            return None
        x_n = x_n - f.evaluate(x_n)/Dfxn
    return None

#Problem 1 Part B
def newton_extremum(f, x0, tol=1e-8):
    """
    find a point x which is close to a local maximum or minimum of f,
    measured by abs(f'(x)) < tol

    Use Newton's method starting at point x0
    """
    x_n = x0
    for n in range(0,10000):
        Dfxn = f.derivative().evaluate(x_n)
        if abs(Dfxn) < tol:
            return x_n
        if Dfxn == 0:
            return 0
        x_n = x_n - f.evaluate(x_n)/Dfxn
    return 0