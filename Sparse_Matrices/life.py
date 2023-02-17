"""
life.py

Put any requested function or class definitions in this file.  You can use these in your script.

Please use comments and docstrings to make the file readable.
"""
import numpy as np
import scipy.sparse as sparse


# From class notes
def neighbors(i, j, m, n):
    """
    gives all the neighbors of the point (i,j) in an (m,n) grid
    """
    # Neighbors to the left/right
    inbrs = [-1, 0, 1]
    
    # Boundary conditions
    if i == 0:
        inbrs = [0, 1]
    if i == m-1:
        inbrs = [-1, 0]
        
    # Neighbors to the top/bottom
    jnbrs = [-1, 0, 1]
    
    # Boundary conditions
    if j == 0:
        jnbrs = [0, 1]
    if j == n-1:
        jnbrs = [-1, 0]

    # Iterator over all possible neighbors of (i,j)
    for delta_i in inbrs:
        for delta_j in jnbrs:
            if delta_i == delta_j == 0:
                continue
            yield i + delta_i, j + delta_j

            
def count_alive_neighbors(S):
    """
    return counts of alive neighbors in the state array S.
    """
    # Initialize cts
    m, n = S.shape
    cts = np.zeros(S.shape, dtype=np.int64)
    
    # Looping over all points in the grid and calculating how many neighbors each one has
    for i in range(m):
        for j in range(n):
            for i2, j2 in neighbors(i, j, m, n):
                cts[i,j] = cts[i,j] + S[i2, j2]

    return cts


# Part B
def grid_adjacency(m,n):
    """
    returns the adjacency matrix for an m x n grid
    """      
    # Initializing adjacency matrix
    mn = m*n
    A = sparse.dok_matrix((mn, mn), dtype=np.float32)
       
    # Looping over one index of A (for example the column index)     
    for k2 in range(mn):
        
        # Finding the equivalent index (i2,j2) of the flat index k2 for an m x n grid
        i2, j2 = np.unravel_index(k2, (m,n))
    
        # Looping over all neighbors of (i2,j2)
        for p, q in neighbors(i2,j2,m,n):
            
            # For each neighbor (p,q), get the flat index k1 that corresponds to it
            k1 = np.ravel_multi_index((p,q),(m,n))
            
            # Update the matrix A
            A[k1,k2] = 1
                          
    
    return A


# Part C
def count_alive_neighbors_matmul(S, A):
    """
    return counts of alive neighbors in the state array S.

    Uses matrix-vector multiplication on a flattened version of S
    """
    # Get size of S
    m, n = S.shape
    
    # flatten S
    s = S.flatten()
    
    # Use adjacency matrix to calculate the counts c as a flat array
    c = A @ s
    
    # Reshape c into a 2d array
    cts = c.reshape(m,n)
    
    return cts



# Part D
def count_alive_neighbors_slice(S):
    """
    return counts of alive neighbors in the state array S.

    Uses slices of cts and S to get final cts
    """
    # Initialize cts
    cts = np.zeros(S.shape)
    
    # Neighbors above
    cts[1:, :] = cts[1:, :] + S[:-1, :]
    
    # Neighbors below
    cts[:-1, :] = cts[:-1, :] + S[1:, :]
    
    # Neighbors left
    cts[:, 1:] = cts[:, 1:] + S[:, :-1]
    
    # Neighbors right
    cts[:, :-1] = cts[:, :-1] + S[:, 1:]
    
    # Neighbors bottom right
    cts[:-1, :-1] = cts[:-1, :-1] + S[1:, 1:]
    
    # Neighbors top left
    cts[1:, 1:] = cts[1:, 1:] + S[:-1, :-1]
    
    # Neighbors top right
    cts[1:, :-1] = cts[1:, :-1] + S[:-1, 1:]
    
    # Neighbors bottom left
    cts[:-1, 1:] = cts[:-1, 1:] + S[1:, :-1]
    
    return cts










