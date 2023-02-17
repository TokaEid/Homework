"""
matlib.py

Put any requested function or class definitions in this file.  You can use these in your script.

Please use comments and docstrings to make the file readable.
"""
import numpy as np
import scipy as sp
import scipy.linalg as la


# Problem 0 Part A
def solve_chol(A,b):
    """
    solve A * x = b for x
    using Chelosky Decomposition
    """
    
    L = la.cholesky(A, lower=True) 
    # gets the lower triangular matrix using built-in chelosky function
    
    x = la.solve_triangular(
            L.T,
            la.solve_triangular(
                L, 
                b, 
                lower=True
            ),
            lower=False
        )    
    # Solves the matrix equation L @ L.T @ x = b by solving the triangular equation L @ y = b first then 
    # solving L.T @ x = y to get x
    
    return x


#Part B
def solve_lu(A, b):
    """
    solve A * x = b for x
    
    use LU decomposition
    """
    P, L, U = la.lu(A)
    x = la.solve_triangular(
            U,
            la.solve_triangular(
                L,
                P.T @ b,
                lower=True
            ),
            lower=False
        )
    # This is the code the professor used in his notes for lU decomposition (I just copied it)
    # Works the same way as above method but with U @ x = y instead of L.T and uses P.T @ b instead of just b
    
    return x
    
    
#Part C
def matrix_pow(A, n):
    """
    solves A**n
    using an eigenvalue decomposition
    """
    
    L, Q = la.eigh(A)
    # Gets lambda (eigenvalues) and Q (orthogonal matrix of eigenvectors)
    
    L = L*np.eye(len(L))
    # Multiplies the list of eigenvalues by the identity matrix to get a diagonal matrix of eigenvalues

    L_n = np.linalg.matrix_power(L, n)
    # Calculates the power of the diagonal matrix L
    
    A_n = Q @ L_n @ Q.T
    # Calculates A**n using the orthogonal matrices Q and Q.T and L**n according to the eigenvalue decomposition method
    
    return A_n


#Part D
def abs_det(A):
    """
    solves det(A)
    using LU decomposition of A = PLU 
    hence det(A) = det(P)*det(L)*det(U)
    """
    
    P, L, U = la.lu(A)
    # Since det(L) = 1 and det(P) = +/-1 then det(A) = +/-det(U)
    # and abs(det(A)) = det(U)
    
    r , c = np.shape(U)
    # Square matrix means r = c
    
    detU = 1
    for i in range(r):
        detU = detU * U[i,i]
        # For each row, multiply the det of U by the element on the diagonal and store it back in detU
    
    return np.absolute(detU)

    
    
# Problem 1 Part A
from numba import njit

# The following functions all follow the same procedure but with the order of the for loops interchanged according to the order
# of the indices in the function's name

@njit
def matmul_ijk(B, C):
    """
    Solves A = B @ C using the order of indices ijk
    """
    r1,c1 = np.shape(B)
    r2,c2 = np.shape(C)
    
    # p and q indicate the size of the final matrix A while r is the index of the most inner loop
    p = r1
    q = c2
    r = c1
    
    # Initializing matrix A
    A = np.zeros((p, q))

    for i in range(p):
            
        for j in range(q):
            
            for k in range(r):
                # Looping over the 3 indices ijk where k is the innermost loop and i is the outermost loop
                # to calculate each element of A
                A[i,j] = A[i,j] + B[i,k] * C[k,j]
        
    return A

@njit
def matmul_ikj(B, C):
    """
    Solves A = B @ C using the order of indices ikj
    """
    r1,c1 = np.shape(B)
    r2,c2 = np.shape(C)

    # p and q indicate the size of the final matrix A while r is the index of the most inner loop
    p = r1
    q = c2
    r = c1

    # Initializing matrix A
    A = np.zeros((p, q))

    for i in range(p):
            
        for k in range(q):
            
            for j in range(r):
                # Looping over the 3 indices ikj where j is the innermost loop and i is the outermost loop
                # to calculate each element of A
                A[i,k] = A[i,k] + B[i,j] * C[j,k]
        
    return A

@njit
def matmul_jik(B, C):
    """
    Solves A = B @ C using the order of indices jik
    """
    r1,c1 = np.shape(B)
    r2,c2 = np.shape(C)
    
    # p and q indicate the size of the final matrix A while r is the index of the most inner loop
    p = r1
    q = c2
    r = c1
    
    # Initializing matrix A
    A = np.zeros((p, q))

    for j in range(p):
            
        for i in range(q):
            
            for k in range(r):
                # Looping over the 3 indices jik where k is the innermost loop and j is the outermost loop
                # to calculate each element of A
                A[j,i] = A[j,i] + B[j,k] * C[k,i]
        
    return A

@njit
def matmul_jki(B, C):
    """
    Solves A = B @ C using the order of indices jki
    """
    r1,c1 = np.shape(B)
    r2,c2 = np.shape(C)
    
    # p and q indicate the size of the final matrix A while r is the index of the most inner loop
    p = r1
    q = c2
    r = c1
    
    # Initializing matrix A
    A = np.zeros((p, q))

    for j in range(p):
            
        for k in range(q):
            
            for i in range(r):
                # Looping over the 3 indices jki where i is the innermost loop and j is the outermost loop
                # to calculate each element of A
                A[j,k] = A[j,k] + B[j,i] * C[i,k]
        
    return A

@njit
def matmul_kij(B, C):
    """
    Solves A = B @ C using the order of indices kij
    """
    r1,c1 = np.shape(B)
    r2,c2 = np.shape(C)

    # p and q indicate the size of the final matrix A while r is the index of the most inner loop    
    p = r1
    q = c2
    r = c1
    
    # Initializing matrix A
    A = np.zeros((p, q))

    for k in range(p):
            
        for i in range(q):
            
            for j in range(r):
                # Looping over the 3 indices kij where j is the innermost loop and k is the outermost loop
                # to calculate each element of A
                A[k,i] = A[k,i] + B[k,j] * C[j,i]
        
    return A

@njit
def matmul_kji(B, C):
    """
    Solves A = B @ C using the order of indices kji
    """
    r1,c1 = np.shape(B)
    r2,c2 = np.shape(C)

    # p and q indicate the size of the final matrix A while r is the index of the most inner loop    
    p = r1
    q = c2
    r = c1
    
    # Initializing matrix A
    A = np.zeros((p, q))

    for k in range(p):
           
        for j in range(q):
            
            for i in range(r):
                # Looping over the 3 indices kji where i is the innermost loop and k is the outermost loop
                # to calculate each element of A
                A[k,j] = A[k,j] + B[k,i] * C[i,j]
        
    return A

# Part B
@njit
def matmul_blocked(B,C):
    """
    Solves A = B @ C using blocked multiplication
    by making use of slices of the matrices and recursion
    """   
    # Using shape of B to initialize matrix A
    n = np.shape(B)[0]  
    A = np.zeros((n, n))
    
    if n <= 64:
        # End condition for recursion, uses previously defined matrix multiplication function    
        A = matmul_kij(B,C)
        
        return A
        
    elif n > 64: 
        # Use slices so that for each recursion, the number of rows and columns goes down by a factor of 2
        slices = (slice(0, n//2), slice(n//2, n))
        
        #Loop over the slices
        for I in slices:
            
            for J in slices:
            
                for K in slices:
                    # recursively call on the function to calculate the multiplication of the sliced (smaller) matrices
                    A[I,J] = A[I,J] + matmul_blocked(B[I,K],C[K,J])
    
        return A
    
    
# Part C
@njit
def matmul_strassen(B,C):
    """
    Solves A = B @ C using Strasen's algorithm for matrix multiplication
    """      
    # Using shape of B to initialize matrix A
    n = np.shape(B)[0]  
    A = np.zeros((n, n))
    
    if n <= 64:
        # End condition for recursion, uses previously defined matrix multiplication function    
        A = matmul_kij(B,C)
        
        return A
        
    elif n > 64:
        
        s1 = slice(0, n//2)
        s2 = slice(n//2, n)

        # Slices the matrices into 4 smaller matrices 
        B11, B12, B21, B22 = B[s1,s1], B[s1,s2], B[s2, s1], B[s2, s2]
        C11, C12, C21, C22 = C[s1,s1], C[s1,s2], C[s2, s1], C[s2, s2]

        # Recursively calls on matmul_strassen to multiply different combinations of the sliced matrices together
        M1 = matmul_strassen((B11 + B22), (C11 + C22))
        M2 = matmul_strassen((B21 + B22), C11)
        M3 = matmul_strassen(B11, (C12 - C22))
        M4 = matmul_strassen(B22, (C21 - C11))
        M5 = matmul_strassen((B11 + B12), C22)
        M6 = matmul_strassen((B21 - B11), (C11 + C12))
        M7 = matmul_strassen((B12 - B22), (C21 + C22))

        # Adds the small matrices together to get the different slices of A
        A[s1, s1] = M1 + M4 - M5 + M7
        A[s1, s2] = M3 + M5
        A[s2, s1] = M2 + M4
        A[s2, s2] = M1 - M2 + M3 + M6
        
        return A




# Problem 2
def markov_matrix(n):
    """
    Finds the doubly stochastic matrix A for a random walk
    with sidewalk length n 
    """  
    # Initialize matrix A
    A = np.zeros((n,n))
    
    for i in range(n):
        
        if i == 0:
            # Boundary condition for the position i = 0 where person either stays in place or moves one step forward
            A[i, i] = 0.5
            A[i+1, i] = 0.5
            
        elif i == n-1:
            # Boundary condition for the position i = n-1 where person either stays in place or moves one step backward
            A[i, i] = 0.5
            A[i-1, i] = 0.5
            
        else:
            # For any other position, probability of moving one step forward or backward is 1/2
            A[i+1, i] = 0.5
            A[i-1, i] = 0.5
       
    return A
    
    
    
    
    
    
    
