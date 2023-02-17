"""
use this file to script the creation of plots, run experiments, print information etc.

Please put in comments and docstrings in to make your code readable
"""
import numpy as np
import scipy as sp
from scipy import sparse
import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from life import neighbors, count_alive_neighbors, grid_adjacency, count_alive_neighbors_matmul, count_alive_neighbors_slice

# Problem 0 Part C and D
def timetest(m,n):
    
    print("Grid Size = (", m, ",", n, ")")
    
    # Initializing S and finding A
    S = np.random.rand(m, n) < 0.3
    A = grid_adjacency(m, n)

    # Time elapsed to run function count_alive_neighbors
    t0 = time.time()
    cts = count_alive_neighbors(S)
    t1 = time.time()
    print("Count_Alive_Neighbors in {} sec.".format(t1 - t0))

    # Changing A to a csc sparse matrix
    A1 = sparse.csc_array(A)

    # Time elapsed to run _matmul function
    t0 = time.time()
    cts = count_alive_neighbors_matmul(S,A1)
    t1 = time.time()
    print("Count_Alive_Neighbors_Matmul Csc Array in {} sec.".format(t1 - t0))

    # Changing A to a csr sparse matrix
    A2 = sparse.csr_array(A)

    # Time elapsed to run _matmul function
    t0 = time.time()
    cts = count_alive_neighbors_matmul(S,A2)
    t1 = time.time()
    print("Count_Alive_Neighbors_Matmul Csr Array in {} sec.".format(t1 - t0))

    # Changing A to a diagonal sparse matrix
    A3 = sparse.dia_array(A)

    # Time elapsed to run _matmul function
    t0 = time.time()
    cts = count_alive_neighbors_matmul(S,A3)
    t1 = time.time()
    print("Count_Alive_Neighbors_Matmul Dia Array in {} sec.".format(t1 - t0))

    # Time elapsed to run _slice function
    # Part D
    t0 = time.time()
    cts = count_alive_neighbors_slice(S)
    t1 = time.time()
    print("Count_Alive_Neighbors_Slice in {} sec.".format(t1 - t0))
    
    return

# Call on timetest function to run for m,n = 100 and m,n = 1000
timetest(100,100)
timetest(1000,1000)


# Part E

# Initializing S
np.random.seed(4)
S = np.random.randn(50,50) < 0.25

# Creating figure
fig = plt.figure(figsize=(5,5))
fig.set_tight_layout(True)

# Plot an image that persists
im = plt.imshow(S, animated=True)
plt.axis('off') # turn off ticks

def update(*args):

    global S
    
    # Update image to display next step using _slice function
    cts = count_alive_neighbors_slice(S)
    # Game of life update
    S = np.logical_or(
        np.logical_and(cts == 2, S),
        cts == 3
    )
    im.set_array(S)
    return im,

# Create animated figure for 50 frames and save it as gif file
anim = FuncAnimation(fig, update, frames=50, interval=200, blit=True)
anim.save('life.gif', dpi=80, writer='imagemagick')






