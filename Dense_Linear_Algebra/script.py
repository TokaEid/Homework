"""
use this file to script the creation of plots, run experiments, print information etc.

Please put in comments and docstrings in to make your code readable
"""
import numpy as np
import scipy as sp
import scipy.linalg as la
from scipy.linalg import blas
import time
import matplotlib.pyplot as plt
from numba import njit

# Problem 0 functions
from matlib import solve_chol, solve_lu, matrix_pow, abs_det
# Problem 1 functions
from matlib import matmul_ijk, matmul_ikj, matmul_jik, matmul_jki, matmul_kij, matmul_kji, matmul_blocked, matmul_strassen
# Problem 2 functions
from matlib import markov_matrix


# Problem 0 Part B
ns = np.round(np.logspace(1,3.301, num=10))
# I found the 3.301 by getting the log_10(2000) which was ~3.301...
ts_chol = []
ts_lu = []

for n in ns:
    
    n = int(n)
    # n needed to be an integer to make the randn function work correctly
    A = np.random.randn(n,n)
    A = A @ A.T
    x = np.random.randn(n)
    b = A @ x
    # I followed the same method that the professor used in class where he generated a random x
    # then used it to calculate b, then pretended to forget x and calculate it again. 

    t0 = time.time()
    x2 = solve_chol(A,b)
    t1 = time.time()
    ts_chol.append(t1-t0)
    # Appends the time t1-t0 to the list ts_chol for each n
    
    t2 = time.time()
    x3 = solve_lu(A,b)
    t4 = time.time()
    ts_lu.append(t4-t2)
    # Appends the time t4-t2 to the list ts_lu for each n

    
# Plots the times for both functions against the list of ns as a log-log plot
plt.figure(0)
plt.plot(ns, ts_chol, label="Cholesky Decomposition")
plt.plot(ns, ts_lu, label="LU Decomposition")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Ns")
plt.ylabel("Times")
plt.legend()
plt.title("Comparison of Run Times")
plt.savefig("Run_Times_P0PartB")
plt.show()



# Problem 1 Part A
# Initiating the times lists and defining the ns list
ns = np.round(np.logspace(1,3.301, num=10))
ts_ijk, ts_ikj, ts_jik, ts_jki, ts_kij, ts_kji = [],[],[],[],[],[]
ts_blas = []
ts_np = []

for n in ns:
    
    n = int(n)
    # Defining the two matrices to be multiplied as row-major arrays
    B = np.array(np.random.randn(n,n), order='C')
    C = np.array(np.random.randn(n,n), order='C')
    
    # Precompile 
    A_ijk = matmul_ijk(B,C)
    A_ikj = matmul_ikj(B,C)
    A_jik = matmul_jik(B,C)
    A_jki = matmul_jki(B,C)
    A_kij = matmul_kij(B,C)
    A_kji = matmul_kji(B,C)

    # Appending time for each function in their respective ts list
    # Numpy_matmul
    t0 = time.time()
    A_np = np.matmul(B,C)
    t1 = time.time()
    ts_np.append(t1-t0)
    
    # Blas
    t0 = time.time()
    A_blas = blas.dgemm(1.0, B, C)
    t1 = time.time()
    ts_blas.append(t1-t0)
    
    # Matmul_ijk
    t0 = time.time()
    A_ijk = matmul_ijk(B,C)
    t1 = time.time()
    ts_ijk.append(t1-t0)
    
    # Matmul_ikj
    t0 = time.time()
    A_ikj = matmul_ikj(B,C)
    t1 = time.time()
    ts_ikj.append(t1-t0)

    # Matmul_jik
    t0 = time.time()
    A_jik = matmul_jik(B,C)
    t1 = time.time()
    ts_jik.append(t1-t0)
    
    # Matmul_jki
    t0 = time.time()
    A_jki = matmul_jki(B,C)
    t1 = time.time()
    ts_jki.append(t1-t0)
    
    # Matmul_kij
    t0 = time.time()
    A_kij = matmul_kij(B,C)
    t1 = time.time()
    ts_kij.append(t1-t0)
    
    # Matmul_kji
    t0 = time.time()
    A_kji = matmul_kji(B,C)
    t1 = time.time()
    ts_kji.append(t1-t0)
    

# Plots the times for all the previous functions against the list of ns as a log-log plot
plt.figure(1)
plt.plot(ns, ts_np, label="Numpy Matmul")
plt.plot(ns, ts_blas, label="BLAS")
plt.plot(ns, ts_ijk, label="matmul_ijk")
plt.plot(ns, ts_ikj, label="matmul_ikj")
plt.plot(ns, ts_jik, label="matmul_jik")
plt.plot(ns, ts_jki, label="matmul_jki")
plt.plot(ns, ts_kij, label="matmul_kij")
plt.plot(ns, ts_kji, label="matmul_kji")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Ns")
plt.ylabel("Times")
plt.legend()
plt.title("Comparison of Run Times")
plt.savefig("Run_Times_P1PartA")
plt.show()




# Part B
# Initializing the ns and ts lists
ns = [2**i for i in range(6,12)]
ts_blocked = []
ts_kij = []

for n in ns:
    # Defining the matrices to be multiplied
    B = np.random.randn(n,n)
    C = np.random.randn(n,n)
    
    # Precompile
    A_blocked = matmul_blocked(B,C)
    A_kij = matmul_kij(B,C)
    
    # Appending time for both functions in their respective ts list
    t0 = time.time()
    A_blocked = matmul_blocked(B,C)
    t1 = time.time()
    ts_blocked.append(t1-t0)
    
    t0 = time.time()
    A_kij = matmul_kij(B,C)
    t1 = time.time()
    ts_kij.append(t1-t0)
    

# Plots the times for the previous functions against the list of ns as a log-log plot
plt.figure(2)
plt.plot(ns, ts_blocked, label="Blocked Matrix Multiplication")
plt.plot(ns, ts_kij, label="Basic Matrix Multiplication")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Ns")
plt.ylabel("Times")
plt.legend()
plt.title("Comparison of Run Times")
plt.savefig("Run_Times_P1PartB")
plt.show()


# Part C
# Initializing the ns and ts lists
ns = [2**i for i in range(6,12)]
ts_strassen = []
ts_blocked = []

for n in ns:
    # Defining the matrices to be multiplied
    B = np.random.randn(n,n)
    C = np.random.randn(n,n)
    
    # Precompile
    A_strassen = matmul_strassen(B,C)
    A_blocked = matmul_blocked(B,C)
    
    # Appending time for both functions in their respective ts list
    t0 = time.time()
    A_strassen = matmul_strassen(B,C)
    t1 = time.time()
    ts_strassen.append(t1-t0)
    
    t0 = time.time()
    A_blocked = matmul_blocked(B,C)
    t1 = time.time()
    ts_blocked.append(t1-t0)
    

# Plots the times for the previous functions against the list of ns as a log-log plot
plt.figure(3)
plt.plot(ns, ts_strassen, label="Strassen Matrix Multiplication")
plt.plot(ns, ts_blocked, label="Blocked Matrix Multiplication")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Ns")
plt.ylabel("Times")
plt.legend()
plt.title("Comparison of Run Times")
plt.savefig("Run_Times_P1PartC")
plt.show()




# Problem 2
n = 50
# n is the total length of sidewalk

ns = np.linspace(0,n-1)
# ns is an array from 0 to n-1

A = markov_matrix(n)
# A is the doubly stochastic matrix with the probabilities

p_t0 = np.zeros(n)
p_t0[0] = 1
# p_t0 is the initial position of the person at time t=0

plt.figure(4)

ts = [10, 100, 1000]
for i in range(len(ts)):
    
    t = ts[i]
    # Raise the matrix A to the power of t and multiply it by the initial position p_t0
    p_t = matrix_pow(A,t) @ p_t0
    
    # Plot p_t for each time in ts
    plt.plot(ns, p_t, label = t)
    plt.legend()
    plt.title("Markov Chain")
    plt.xlabel("Position")
    plt.ylabel("Probability Distribution")
    

plt.savefig("Problem2")    
plt.show()

# Get eigenvalues and eigenvectors of A    
Lambda, v = la.eigh(A)

# Find the index of the largest eigenvalue in Lambda and get the eigenvector corresponding to it
max_val_idx = np.argmax(Lambda)
v = v[max_val_idx]

# Normalize v 
v = v / np.sum(v)

# Check that vectors are the same shape
if np.shape(v) != np.shape(p_t):
    raise "Vectors are not the same shape"
    
# Calculate Euclidean distance between v and p_t at t=1000    
euc_dist = la.norm(p_t-v)
print("Euclidean distance between the eigenvector v and p_t at t=1000 is ", euc_dist)
    
# Repeat above steps for t=2000    
t = 2000
p_t2 = matrix_pow(A,t) @ p_t0

if np.shape(v) != np.shape(p_t2):
    raise "Vectors are not the same shape"

euc_dist2 = la.norm(p_t2-v)
print("Euclidean distance between the eigenvector v and p_t at t=2000 is ", euc_dist2)




