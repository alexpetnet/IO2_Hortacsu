# Setup
import numpy as np
import scipy as sp
from scipy.optimize import fsolve
np.random.seed(1234)

# Equilibrium conditions for a BNE
def bne(p,α,δ,x):
    f1 = p[0] - np.exp(α*x - δ*p[1])/(1+np.exp(α*x - δ*p[1]))
    f2 = p[1] - np.exp(α*x - δ*p[0])/(1+np.exp(α*x - δ*p[0]))
    return np.array([f1,f2])

# Solving the fixed points for BNE with a variety of start points
def solve_bne(α,δ,x):
    bne_p = lambda p: bne(p,α,δ,x)
    solve = lambda x0: fsolve(bne_p,x0)
    grid = np.mgrid[0:1:1e-1, 0:1:1e-1].reshape(2,-1).T
    sols = np.round(np.apply_along_axis(solve, 1, grid),6)
    return np.unique(sols, axis=0)

# Compute for the given params
print(solve_bne(1,1,1))
print(solve_bne(3,6,1))
print(solve_bne(1,1,2))
print(solve_bne(3,6,2))


# Generate sample using the symmetric selection rule
def gen_sym_sample(α,δ,T,S):
    '''Y1 and Y2 are (T,S) shaped arrays of decisions for each firm in each
    market, x is an (T,) shaped array of the given variable'''
    ε = np.random.logistic(loc=0,scale=1,size=(T,S,2))
    X = 1+np.random.binomial(n=1,p=.5,size=(T,S))
    sol1 = solve_bne(α,δ,1)[1,0]
    sol2 =  solve_bne(α,δ,2)[1,0]
    Y1 = (((X==1)*(α*X-δ*sol1)+(X==2)*(α*X-δ*sol1))<=ε[:,:,0])+0
    Y2 = (((X==1)*(α*X-δ*sol1)+(X==2)*(α*X-δ*sol1))<=ε[:,:,1])+0
    return Y1, Y2, X

# Solve bne under symmetry assumption

# Log likelihood given data and proposed equilibrium probabilities

# Maximise the log likelihood as the solution to a fixed point problem
def sym_mle(Y1,Y2,X,α0,δ0):
    ''' α0 and δ0 are the initial candidate parameter values'''
    return α,δ


 Y1,Y2,X = gen_sym_sample(3,6,1000,50)



# Generate sample using the probabilistic selection rule
def gen_prob_sample(α,δ,T,S):
    '''Y1 and Y2 are (T,S) shaped arrays of decisions for each firm in each
    market, X is a (T,S) shaped array of the given variable'''
    return Y1, Y2, X
