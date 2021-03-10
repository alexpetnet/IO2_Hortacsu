import numpy as np
import pandas as pd

from scipy.optimize import minimize

data = pd.read_csv('data/psetTwo.csv')

K = 10
β = .999

M = data.iloc[:, 0]

d = np.array([int(M[i] < M[i - 1]) for i in range(1, len(M))])
d_p = np.insert(d, 0, 0) # need to subset those rows that are after changing engine
d = np.append(d, 0) # this is the replacement variable

cap = max(M)
X = np.arange(cap, step = cap / K)
X




mat = np.array([[int(X[i - 1] <= M[j] < X[i]) for i in range(1, len(X))] for j in range(len(M))])
lc = np.array([[int(M[j] > X[len(X) - 1])] for j in range(len(M))])
mat = np.concatenate((mat, lc), axis = 1)
mat

# the transition matrix for d = 1
T1 = np.sum(mat[d_p == 1], axis = 0) / np.sum(mat[d_p == 1])
T1 = np.tile(T1, (len(X), 1))

# the transition matrix for d = 0
T0 = np.zeros((len(X), len(X))) #initialising
for j in range(len(X)):

    tr = np.array([(r[j] == 1) for r in mat])
    tr[d == 1] = False
    tr[len(M) - 1] = False
    tr = np.roll(tr, 1)

    T0[j] = np.sum(mat[tr], axis = 0) / np.sum(mat[tr])

T0

### 2.3.1.3

def u0(x, θ1, θ2):
    return - θ1 * x - θ2 * ((x / 100) ** 2)

def u1(x, θ3):
    return - θ3


V_0 = np.zeros(K)

def FP(V_0, θ, T0, T1, ϵ):

    θ1 = θ[0]
    θ2 = θ[1]
    θ3 = θ[2]

    V_old = V_0

    U0 = np.array([u0(x, θ1, θ2) for x in X])
    C0 = β * V_old

    U1 = np.array([u1(x, θ3) for x in X])
    C1 = β * np.repeat(V_old[0], K)

    inner_sum = np.exp(U0 + C0) + np.exp(U1 + C1)
    V_new = T0 @ np.log(inner_sum)

    while max(np.abs(V_new - V_old)) > ϵ:
        V_old = V_new

        C0 = β * V_old

        C1 = β * np.repeat(V_old[0], K)

        inner_sum = np.exp(U0 + C0) + np.exp(U1 + C1)
        #print(inner_sum)
        V_new = T0 @ np.log(inner_sum)
        #print(V_new)

    return V_new

# to addreess the nan issue, increase the precision of EV
def FP_longdouble(V_0, θ, T0, T1, ϵ):

    θ1 = θ[0]
    θ2 = θ[1]
    θ3 = θ[2]

    V_old = V_0

    U0 = np.array([u0(x, θ1, θ2) for x in X])
    C0 = β * V_old

    U1 = np.array([u1(x, θ3) for x in X])
    C1 = β * np.repeat(V_old[0], K)

    inner_sum = np.exp(np.longdouble(U0 + C0)) + np.exp(np.longdouble(U1 + C1))
    V_new = T0 @ np.log(inner_sum)

    while max(np.abs(V_new - V_old)) > ϵ:
        V_old = V_new

        C0 = β * V_old

        C1 = β * np.repeat(V_old[0], K)

        inner_sum = np.exp(np.longdouble(U0 + C0)) + np.exp(np.longdouble(U1 + C1))
        #print(inner_sum)
        V_new = T0 @ np.log(inner_sum)
        #print(V_new)

    return V_new


v = FP(V_0, np.array([.01, .01, .01]), T0, T1, 0.00001)
v

def J(θ, V, T0):

    θ1 = θ[0]
    θ2 = θ[1]
    θ3 = θ[2]

    U0 = np.array([u0(x, θ1, θ2) for x in X])
    U1 = np.array([u1(x, θ3) for x in X])

    pk = 1 / (1 + np.exp(U1 + β * np.repeat(V[0], K) - U0 - β * V))

    J = T0 @ np.diag(pk) * β

    J[:, 0] = J[:, 0] + T0 @ (1 - pk) * β

    return J


Jac = J(np.array([.01, .01, .01]), v, T0)
Jac

def newton(V0, θ, ϵ):

    θ1 = θ[0]
    θ2 = θ[1]
    θ3 = θ[2]

    V_old = V0

    U0 = np.array([u0(x, θ1, θ2) for x in X])
    C0 = β * V_old

    U1 = np.array([u1(x, θ3) for x in X])
    C1 = β * np.repeat(V_old[0], K)

    inner_sum = np.exp(U0 + C0) + np.exp(U1 + C1)
    ΓV = T0 @ np.log(inner_sum)

    Jac = J(θ, V_old, T0)

    V_new = V_old - np.linalg.inv(np.eye(K) - Jac) @ (V_old - ΓV)
    #V_new = V_old - np.linalg.solve(np.eye(K) - Jac, V_old - ΓV)

    while max(np.abs(V_new - V_old)) > ϵ:
        V_old = V_new

        C0 = β * V_old

        C1 = β * np.repeat(V_old[0], K)

        inner_sum = np.exp(U0 + C0) + np.exp(U1 + C1)
        ΓV = T0 @ np.log(inner_sum)

        Jac = J(θ, V_old, T0)

        V_new = V_old - np.linalg.inv(np.eye(K) - Jac) @ (V_old - ΓV)
        #V_new = V_old - np.linalg.solve(np.eye(K) - Jac, V_old - ΓV)

    return V_new

newton(V_0, np.array([.1, .1, .1]), np.power(0.1, 10))


def poly(V0, θ, ϵ_fp, ϵ_n):

    EV_fp = FP(V0, θ, T0, T1, ϵ_fp)

    EV_n = newton(EV_fp, θ, ϵ_n)

    return EV_n

poly(V_0, np.array([.1, .1, .1]), np.power(0.1, 10), np.power(0.1, 16))


### 2.3.1.4

def l(θ, m, d, V, T0, T1):

    θ1 = θ[0]
    θ2 = θ[1]
    θ3 = θ[2]

    ind = np.array([X[i - 1] <= m < X[i] for i in range(1, len(X))])
    lc = m >= X[len(X) - 1]
    ind = np.append(ind, lc)

    l0 = np.exp(u0(m, θ1, θ2) + β * V[ind])
    l1 = np.exp(u1(m, θ3) + β * V[0])

    p0 = l0 / (l0 + l1)

    return (1 - d) * np.log(p0) + d * np.log(1 - p0)



def dl(θ, m, d, V, dp):

    θ1 = θ[0]
    θ2 = θ[1]
    θ3 = θ[2]

    ind = np.array([X[i - 1] <= m < X[i] for i in range(1, len(X))])
    lc = m >= X[len(X) - 1]
    ind = np.append(ind, lc)

    l0 = np.exp(u0(m, θ1, θ2) + β * V[ind])
    l1 = np.exp(u1(m, θ3) + β * V[0])

    p0 = l0 / (l0 + l1)

    return (d / (1 - p0)) * (- dp[ind, :]) + ((1 - d) / p0) * dp[ind, :]


dl(np.array([.01, .01, .01]), 1, 1, v, Jac)


def L(θ, M, D, ld):

    # this is to choose whether to use increased precision or not
    if ld == True:
        V = FP_longdouble(V_0, θ, T0, T1, np.power(0.1, 10))
    else:
        V = FP(V_0, θ, T0, T1, np.power(0.1, 10))
    #V = FP(V_0, θ, T0, T1, 0.000001)
    #V = poly(V_0, θ, np.power(0.1, 10), np.power(0.1, 16))
    #V = FP_longdouble(V_0, θ, T0, T1, np.power(0.1, 10))

    lik = np.array([l(θ, M[i], D[i], V, T0, T1) for i in range(len(M))])


    Lik = - np.sum(lik)
    print(Lik)
    return Lik


L(np.array([1, 1, 1]), M, d, False)


### 2.3.1.5

def dL(θ, M, D):

    θ1 = θ[0]
    θ2 = θ[1]
    θ3 = θ[2]

    V = FP(V_0, θ, T0, T1, np.power(0.1, 10))
    #V = poly(V_0, θ, np.power(0.1, 10), np.power(0.1, 16))
    #V = FP_longdouble(V_0, θ, T0, T1, np.power(0.1, 10))

    U0 = np.array([u0(x, θ1, θ2) for x in X])
    U1 = np.array([u1(x, θ3) for x in X])

    pk = 1 / (1 + np.exp(U1 + β * np.repeat(V[0], K) - U0 - β * V))

    u0d = np.array([- X, - (X / 100) ** 2, np.zeros(K)]).T
    u1d = np.array([np.zeros(K), np.zeros(K), - np.ones(K)]).T

    l0 = np.exp(U0 + β * V)
    l1 = np.exp(U1 + β * np.repeat(V[0], K))
    denom = l0 + l1
    nom = l0[:, None] * u0d + l1[:, None] * u1d
    dΓ = T0 @ (nom / denom[:, None])


    Jac = J(θ, V, T0)

    dEV = np.linalg.inv(np.eye(K) - Jac) @ dΓ

    dp = - np.multiply(pk ** 2, l1 / l0)[:, None] * (u1d + β * np.tile(dEV[0, :], (len(X), 1)) -
        u0d - β * dEV)

    D = np.array([dl(θ, M[i], D[i], V, dp) for i in range(len(M))])

    return - np.sum(D, axis = 0)[0]

out = dL(np.array([.01, .01, .01]), M, d)
out


### 2.3.1.6

st = np.zeros(3)
opt = minimize(L, x0 = st, args = (M, d, False), method = 'BFGS', jac = dL)
print("The estimated parameters are " + str(opt.x))

# optimizing without gradient but with increased precision
# doesn't really help
opt_pr = minimize(L, x0 = st, args = (M, d, True), method = 'BFGS', options = {'disp' : True})
opt_pr.x
#L(opt_pr.x, M, d)






##### JAX #####

from jax import grad
import jax.numpy as jnp

θ_check = jnp.ones(3) * 0.01

M_jax = jnp.array(M, dtype = float)
M_jax
d_jax = jnp.array(d, dtype = float)
d_jax

V_0_jax = jnp.zeros(K)

def FP_jax(V_0, θ, T0, T1, ϵ):

    θ1 = θ[0]
    θ2 = θ[1]
    θ3 = θ[2]

    V_old = V_0

    U0 = np.array([u0(x, θ1, θ2) for x in X])
    C0 = β * V_old

    U1 = np.array([u1(x, θ3) for x in X])
    C1 = β * np.repeat(V_old[0], K)

    inner_sum = jnp.exp(U0 + C0) + jnp.exp(U1 + C1)
    V_new = T0 @ np.log(inner_sum)

    while max(np.abs(V_new - V_old)) > ϵ:
        V_old = V_new

        C0 = β * V_old

        C1 = β * np.repeat(V_old[0], K)

        inner_sum = jnp.exp(U0 + C0) + jnp.exp(U1 + C1)
        #print(inner_sum)
        V_new = T0 @ np.log(inner_sum)
        #print(V_new)

    return V_new

def poly_jax(V0, θ, ϵ_fp, ϵ_n):

    EV_fp = FP_jax(V0, θ, T0, T1, ϵ_fp)

    EV_n = newton(EV_fp, θ, ϵ_n)

    return EV_n


def L_jax(θ):

    V = FP_jax(V_0_jax, θ, T0, T1, 0.000001)
    #V = poly_jax(V_0, θ, np.power(0.1, 10), np.power(0.1, 16))
    #V = newton(V_0, θ, np.power(0.1, 10))

    lik = np.array([l(θ, M_jax[i], D_jax[i], V, T0, T1) for i in range(len(M_jax))])

    #for i in range(len(M)):
    #    if lik[i] == 0:
    #        print("0")

    #product = 1
    #for i in range(len(M)):
    #    product = product * lik[i]

    return - np.sum(lik)
    #return float(product)

dL_jax = grad(L_jax)
dL_jax(θ_check)
