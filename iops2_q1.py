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
    ''' note that relative to the pset notation for increasing probabilities in
    k, the 0th column is player 2 and the 1st column is player 1'''
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
print("Starting First Monte Carlo")
S = 50
results = np.zeros((S,2))
for i in range(S):
    Y1,Y2,X = gen_sym_sample(3,6,1000)
    θ = sym_mle(Y1,Y2,X,1,1)
    results[i,:]= θ

plt.hist(results[:,0], density=True, label="Histogram")
plt.xlabel("α")
plt.ylabel("Density")
plt.savefig("figs/hist_alpha_det.png")
plt.close()


plt.hist(results[:,1], density=True, label="Histogram")
plt.xlabel("δ")
plt.ylabel("Density")
plt.savefig("figs/hist_delta_det.png")
plt.close()



# 1.3.4 ------------------------------------------------------------------------
# Equilibrium probabilities
def eq_probs(K):
    v = np.array([np.exp((i+1)/2) for i in range(K)])
    return v/np.sum(v)

# Equilibrium selection mechanism
def sel_eq(λ):
    return np.random.choice(np.array(list(range(len(λ)))),p=λ)

# Generate sample using the probabilistic selection rule
def gen_prob_sample(α,δ,T):
    '''Y1 and Y2 are (T,S) shaped arrays of decisions for each firm in each
    market, X is a (T,S) shaped array of the given variable
    Note: relative to pset notation for increasing probabilities,
    player numbers are flipped.'''
    ε = np.random.logistic(loc=0,scale=1,size=(T,2))
    X = 1+np.random.binomial(n=1,p=.5,size=(T,))
    p1,p2 = solve_bne(α,δ,1), solve_bne(α,δ,2)
    Y1,Y2 = np.zeros(T),np.zeros(T)
    for t in range(T):
        if X[t]==1:
            p = p1[sel_eq(eq_probs(3)),:]
        else:
            p = p2[sel_eq(eq_probs(3)),:]
        Y1[t] = (α*X[t] - δ*p[1] >= -ε[t,0])+0
        Y2[t] = (α*X[t] - δ*p[0] >= -ε[t,1])+0
    return Y1, Y2, X

# Return equilibrium probabilities given realised X and conjectured α,δ
def prob_sel(α,δ):
    p1 = solve_bne(α,δ,1)
    p2 = solve_bne(α,δ,2)
    return p1,p2

# Minus Log likelihood given data, and conjectured probabilities and
# equilibrium selection
def prob_llik(α,δ,Y1,Y2,X,p1,p2):
    ''' P is (K,T,2): number of equilibria x markets x players'''
    K1,K2 = p1.shape[0],p2.shape[0]
    λ1,λ2 = eq_probs(K1),eq_probs(K2)
    T = len(X)
    # player 1 sums
    p111 = np.array([λ1[i]*(1- sps.logistic.cdf(δ*p1[i,1]-α*X)) \
    for i in range(K1)])
    p112 =np.array([λ2[i]*(1- sps.logistic.cdf(δ*p2[i,1]-α*X)) \
    for i in range(K2)])
    p11 = np.sum([X==1]*p111,axis=0) + np.sum([X==2]*p112,axis=0)
    # player 2 sums
    p211 = np.array([λ1[i]*(1- sps.logistic.cdf(δ*p1[i,0]-α*X)) \
    for i in range(K1)])
    p212 =np.array([λ2[i]*(1- sps.logistic.cdf(δ*p2[i,0]-α*X)) \
    for i in range(K2)])
    p21 = np.sum([X==1]*p211,axis=0) + np.sum([X==2]*p212,axis=0)
    # combine into likelihood
    ll = Y1*np.log(p11)+ (1-Y1)*np.log(1-p11) + \
    Y2*np.log(p21)+ (1-Y2)*np.log(1-p21)
    return -np.sum(ll)

# MLE estimator
def prob_mle(Y1,Y2,X,α0,δ0):
    ''' α0 and δ0 are the initial candidate parameter values'''
    iter = 0
    diff = 1
    θo = np.array([α0,δ0])
    while (iter < 100)&(diff>1e-6):
        p1, p2 = prob_sel(θo[0],θo[1])
        sll = lambda φ: prob_llik(φ[0],φ[1],Y1,Y2,X,p1,p2)
        res = sp.optimize.minimize(sll,θo)
        liknew, θn = res.fun, res.x
        iter+=1
        diff = np.sum(np.abs(θn-θo))
        θo=θn
    return θn

# Conduct a Monte Carlo study and save the results as a histogram
print("Starting second Monte Carlo")
S = 50
results2 = np.zeros((S,2))
for i in range(S):
    print(i)
    Y1,Y2,X = gen_prob_sample(3,6,1000)
    θ = prob_mle(Y1,Y2,X,1,2)
    results2[i,:]= θ

plt.hist(results2[:,0], density=True, label="Histogram")
plt.xlabel("α")
plt.ylabel("Density")
plt.savefig("figs/hist_alpha_prob.png")
plt.close()


plt.hist(results2[:,1], density=True, label="Histogram")
plt.xlabel("δ")
plt.ylabel("Density")
plt.savefig("figs/hist_delta_prob.png")
plt.close()

# 1.3.6 Extra analysis ---------------------------------------------------------

# Conduct a Monte Carlo study using part 3 algorithm on part 4 data
print("Starting third Monte Carlo")
S = 50
results3 = np.zeros((S,2))
for i in range(S):
    Y1,Y2,X = gen_prob_sample(3,6,1000)
    θ = sym_mle(Y1,Y2,X,1,2)
    results3[i,:]= θ

plt.hist(results3[:,0], density=True, label="Histogram")
plt.xlabel("α")
plt.ylabel("Density")
plt.savefig("figs/hist_alpha_det_on_prob.png")
plt.close()


plt.hist(results3[:,1], density=True, label="Histogram")
plt.xlabel("δ")
plt.ylabel("Density")
plt.savefig("figs/hist_delta_det_on_prob.png")
plt.close()

# Conduct a Monte Carlo study using part 4 algorithm on part 3 data
print("Starting fourth Monte Carlo")
S = 50
results4 = np.zeros((S,2))
for i in range(S):
    Y1,Y2,X = gen_sym_sample(3,6,1000)
    θ = prob_mle(Y1,Y2,X,1,1)
    results4[i,:]= θ

plt.hist(results4[:,0], density=True, label="Histogram")
plt.xlabel("α")
plt.ylabel("Density")
plt.savefig("figs/hist_alpha_prob_on_det.png")
plt.close()


plt.hist(results4[:,1], density=True, label="Histogram")
plt.xlabel("δ")
plt.ylabel("Density")
plt.savefig("figs/hist_delta_prob_on_det.png")
plt.close()
