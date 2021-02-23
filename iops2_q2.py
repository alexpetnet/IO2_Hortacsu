import numpy as np
import pandas as pd

data = pd.read_csv('data/psetTwo.csv')

data
K = 100
β = .999

V_0 = np.zeros(K)

# need transition matrices
def FP(V_0, θ, T0, T1, ϵ):

    θ1 = θ[0]
    θ2 = θ[1]
    θ3 = θ[2]

    V_old = V_0

    U0 = np.array([u0(x, θ1, θ2) for x in X])
    C0 = β * T0 @ V_0

    U1 = np.array([u1(x, θ3) for x in X])
    C1 = β * T1 @ V_0

    inner_sum = np.exp(U0 + C0) + np.exp(U1 + C1)
    V_new = 0.577 + np.log(inner_sum)

    while max(np.abs(V_new - V_old)) > ϵ:
        V_old = V_new

        U0 = np.array([u0(x, θ1, θ2) for x in X])
        C0 = β * T0 @ V_0

        U1 = np.array([u1(x, θ3) for x in X])
        C1 = β * T1 @ V_0

        inner_sum = np.exp(U0 + C0) + np.exp(U1 + C1)
        V_new = 0.577 + np.log(inner_sum)

    return V_new

# T0 - the right row in the transition matrix
def l(θ, m, d, V, T0, T1):

    θ1 = θ[0]
    θ2 = θ[1]
    θ3 = θ[2]

    l0 = np.exp(u0(m, θ1, θ2) + β * T0 @ V)
    l1 = np.exp(u1(m, θ3) + β * T1 @ V)
    sum = l0 + l1

    if d = 0:
        return l0 / sum

    if d = 1:
        return l1 / sum

def L(θ, M, D):

    V = FP(V_0, θ, T0, T1, ϵ)

    return np.prod([l(θ, M[i], D[i], V, t0, t1) for i in range(len(M))])
