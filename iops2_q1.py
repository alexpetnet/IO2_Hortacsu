# Setup
import numpy as np
import scipy as sp
import scipy.stats as sps
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
np.random.seed(1234)

# 1.3.2 ------------------------------------------------------------------------
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

# Same restricted to the symmetric case
def sym_bne(p,α,δ,x):
    return p - np.exp(α*x - δ*p)/(1+np.exp(α*x - δ*p))

def solve_sym_bne(α,δ,x):
    solve = lambda p :sym_bne(p,α,δ,x)
    return fsolve(solve,0)[0]

# Tell us the answers
for z in [(1,1,1),(3,6,1),(1,1,2),(3,6,2)]:
    print("The equilibrium probabilities for (α,δ,x)=" + str(z) + " are " +\
    str(solve_bne(*z)))

# 1.3.3 ------------------------------------------------------------------------
# Generate sample using the symmetric selection rule
def gen_sym_sample(α,δ,T):
    '''Y1 and Y2 are (T,) shaped arrays of decisions for each firm in each
    market, X is also a (T,) array for the realised Bernoulli variable'''
    ε = np.random.logistic(loc=0,scale=1,size=(T,2))
    X = 1+np.random.binomial(n=1,p=.5,size=(T,))
    p1,p2 = solve_sym_bne(α,δ,1), solve_sym_bne(α,δ,2)
    Y1 = (((X==1)*(α*X-δ*p1)+(X==2)*(α*X-δ*p2))>=-ε[:,0])+0
    Y2 = (((X==1)*(α*X-δ*p1)+(X==2)*(α*X-δ*p2))>=-ε[:,1])+0
    return Y1, Y2, X

# Probabilities across markets in the symmetric case
def sym_prob(α,δ,X):
    p1,p2 = solve_sym_bne(α,δ,1), solve_sym_bne(α,δ,2)
    return p1*(X==1) + p2*(X==2)

# Minus Log likelihood given data and proposed equilibrium probabilities
def sym_llik(α,δ,Y1,Y2,X,P):
    F = sps.logistic.cdf(δ*P-α*X)
    return -np.sum((Y1+Y2)*np.log(1-F) + (2-Y1-Y2)*np.log(F))

# Maximise the log likelihood as the solution to a fixed point problem
def sym_mle(Y1,Y2,X,α0,δ0):
    ''' α0 and δ0 are the initial candidate parameter values'''
    iter = 0
    diff = 1
    θo = np.array([α0,δ0])
    while (iter < 1000)&(diff>1e-6):
        P = sym_prob(θo[0],θo[1],X)
        sll = lambda φ: sym_llik(φ[0],φ[1],Y1,Y2,X,P)
        res = sp.optimize.minimize(sll,θo)
        liknew, θn = res.fun, res.x
        iter+=1
        diff = np.sum(np.abs(θn-θo))
        θo=θn
    return θn


# Conduct a Monte Carlo study and save the results as a histogram
results = np.zeros(50,2)

Y1,Y2,X = gen_sym_sample(3,6,1000)
sym_mle(Y1,Y2,X,1,1)


# 1.3.4 ------------------------------------------------------------------------
# Generate sample using the probabilistic selection rule
def gen_prob_sample(α,δ,T,S):
    '''Y1 and Y2 are (T,S) shaped arrays of decisions for each firm in each
    market, X is a (T,S) shaped array of the given variable'''
    return Y1, Y2, X
