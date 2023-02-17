"""
fibonacci

functions to compute fibonacci numbers

Complete problems 2 and 3 in this file.
"""

import time # to compute runtimes
from tqdm import tqdm # progress bar
import matplotlib.pyplot as plt


# Question 2
def fibonacci_recursive(n):
    """
    This function recursively adds the element at n-1 to the element at n-2 until n reaches 0 and 1 (the endpoints)
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

if __name__ == '__main__':
    # this code runs when executed as a script
    print("Fibonacci Recursive Sequence:")
    for n in range(30):
        print(fibonacci_recursive(n))


        
# Question 2
def fibonacci_iter(n):
    """
    This function iteratively goes over every number from 0 to n-1 then saves the value of 'b' in 'a' and saves the 
    new added sum (of a and b) in 'b' and keeps going iteratively n-1 times and returns the last sum 'b'
    """
    a = 0
    b = 1
    if n == 0:  #Important condition to make sure n=0 gives the correct answer
        return a 
    else:    
        for i in range(n-1):
            a,b = b,a+b
        
    return b

if __name__ == '__main__':
    # this code runs when executed as a script
    print("Fibonacci Iterative Sequence:")
    for n in range(30):
        print(fibonacci_iter(n))

        
        
# Question 3
import numpy as np
import sys

sys.setrecursionlimit(10000) 

# I was running into two problems: (1) RuntimeWarning: overflow encountered in matmul and (2) RecursionError: maximum recursion depth 
# exceeded while calling a Python object
# To solve the first problem, I tried changing the dtype of each array and using int64 seemed to be the only one that ran without
# warning
# To fix the second problem, I had to change the recursion limit to 10000 so the code would stop crashing

def fibonacci_power(n):
    """
    Uses matrix multiplication to evaluate powers of a matrix to get the Fibonacci sequence using
    x_n = A^n-1 x_1 where F_n = x_n(0)
    
    """
    
    A = np.array([[1.,1.],[1.,0.]], dtype=np.int64) #Matrix A was defined in the problem
    
    def matrix_power(A,n):
        """
        Computes the power A ** n by matrix multiplying A to itself then recursively matrix multiplying A^2
        with A^n-2
        """
        
        if n < 0: 
            #Condition in place to avoid infinite loops if negative numbers are encountered
            return np.zeros((2,2), dtype=np.int64)
        elif n == 1:
            #End condition when raised to the power of 1
            return A
        elif n == 0:
            #End condition when raised to the power of 0
            return np.eye(2, dtype=np.int64)
        else:
            return np.matmul(np.matmul(A,A), matrix_power(A, n - 2))

    x1 = np.array([1,0]) #Vector x1 defined in the problem
    
    x_n = matrix_power(A,n-1) #x_n here is not the same as defined in problem. Here it only represents A^n-1
    
    F_n = np.matmul(x_n, x1)
    
    return F_n[0]  #Outputs the zeroth element of F_n  (the fibonacci number)


if __name__ == '__main__':
    # this code runs when executed as a script
    print("Fibonacci Power Sequence:")
    for n in range(30):
        print(fibonacci_power(n))



if __name__ == '__main__':
    """
    this section of the code only executes when
    this file is run as a script.
    """
    def get_runtimes(ns, f):
        """
        get runtimes for fibonacci(n)

        e.g.
        trecursive = get_runtimes(range(30), fibonacci_recusive)
        will get the time to compute each fibonacci number up to 29
        using fibonacci_recursive
        """
        ts = []
        for n in tqdm(ns):
            t0 = time.time()
            fn = f(n)
            t1 = time.time()
            ts.append(t1 - t0)

        return ts


    nrecursive = range(35)
    trecursive = get_runtimes(nrecursive, fibonacci_recursive)

    niter = range(10000)
    titer = get_runtimes(niter, fibonacci_iter)

    npower = range(10000)
    tpower = get_runtimes(npower, fibonacci_power)

#Question 4

    # Defining the lists for each ns array
    nrecursive_range = np.arange(35).tolist()
    niter_range = np.arange(10000).tolist()
    npower_range = np.arange(10000).tolist()
       
    # Plotting each set of data separately with their own labels for the legend and adjusting the labels and scales
    # of the x and y axes
    plt.plot(trecursive, nrecursive_range, label="Recursive Function")
    plt.plot(titer, niter_range, label="Iterative Function")
    plt.plot(tpower, npower_range, label="Power Function")
    plt.xlabel("Run Times")
    plt.ylabel("Numbers in Sequence")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("Comparison of Run Times")
    plt.savefig("fibonacci_runtime.png") #Saves image as png file in current directory

    plt.show() 
    
    # For some reason, running this in the terminal didn't show a plot, but it did save it in the directory.
    # However, copying this whole code into a Jupyter ipynb did work and the plot was shown when I ran the code.
    # It might be a problem with my terminal, but I'm not really sure and since it worked in the ipynb, I'm assuming the code is
    # not the problem.




