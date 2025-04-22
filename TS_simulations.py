import scipy.special as sc
from scipy.stats import multivariate_normal
from scipy.stats import multinomial
import numpy as np
import itertools
import math

# Auxiliary computation function
def b(x,comb,jj):
    return sc.beta(x[2*jj]+sum([x[2*j] for j in comb]),x[2*jj+1]+sum([x[2*j+1] for j in comb]))/(sc.beta(sum([x[2*j] for j in comb]),sum([x[2*j+1] for j in comb]))*sc.beta(x[2*jj],x[2*jj+1]))

# Runs one incremental update of the exact value of the posterior probaility and its helper values
def update_full(x,hist,j,s):
    k = len(x)//2
    klist = [i for i in range(k)]
    for level in range(1,k-1):
        for comb in itertools.combinations(klist, level):
            if j in comb:
                hist[comb] += ((-1)**(s+1))*sum([b(x,comb,jj)*hist[tuple(sorted((jj,)+comb))]/sum([x[2*jjj+s] for jjj in comb]) for jj in klist if jj not in comb])
            else:
                hist[comb] += ((-1)**s)*b(x,comb,j)*hist[tuple(sorted((j,)+comb))]/x[2*j+s]
    for comb in itertools.combinations(klist, k-1):
        if j in comb:
            hist[comb] += ((-1)**(s+1))*sum([b(x,comb,jj)/sum([x[2*jjj+s] for jjj in comb]) for jj in klist if jj not in comb])
        else:
            hist[comb] += ((-1)**s)*b(x,comb,j)/x[2*j+s]
    x[2*j+s] += 1
    return x,hist

# Loads initial helper posterior probabilities
def load_hist(k):
    hist = {}
    klist = [i for i in range(k)]
    temp = 1/k
    for comb in itertools.combinations(klist, 1):
        hist[comb] = temp
    for level in range(2,k):
        kk = k - level
        temp = 1/(kk+1) + kk*sum([sc.beta(kk+j,j+2)/(j*sc.beta(j,j+1))-sc.beta(kk+j,j+1)/(j*sc.beta(j,j)) for j in range(1,level)])
        for comb in itertools.combinations(klist, level):
            hist[comb] = temp
    return hist

# Runs a TS trial with burn-in using exact probability computations
def TS_exact(N,p,B,b):
    k = len(p)
    x = [1 for i in range(2*k)]
    hist = load_hist(k)
    for i in range(B):
        for j in range(k):
            s = np.random.binomial(n=1,p=p[j])
            x,hist = update_full(x,hist,j,s)
    update = [hist[(i,)] for i in range(k)]
    for i in range(0,N-k*B,b):
        for j in range(min(N-k*B-i,b)):
            arm = np.random.choice(k, p=update)
            s = np.random.binomial(n=1,p=p[arm])
            x,hist = update_full(x,hist,arm,s)
        update = [hist[(i,)] for i in range(k)]
    for i in range(len(x)):
        x[i] -= 1
    return x

# Calculates GA approximation of posterior probability
def ga(x):
    k = len(x)//2
    mu = []
    sigma = []
    for i in range(k):
        mu.append((x[2*i])/(x[2*i]+x[2*i+1]))
        sigma.append(((x[2*i])*(x[2*i+1]))/((x[2*i]+x[2*i+1]+1)*((x[2*i]+x[2*i+1])**2)))
    out = [multivariate_normal.cdf([0 for j in range(k-1)],mean=[mu[i]-mu[j] for j in range(k) if j!=i],cov=[[sigma[i]+(j==jj)*sigma[j] for jj in range(k) if jj!=i] for j in range(k) if j!=i]) for i in range(k)]
    total = sum(out)
    return [out[i]/total for i in range(k)]

# Runs a TS trial with burn-in using GA approximation
def TS_ga(N,p,B,b):
    k = len(p)
    x = [1 for i in range(2*k)]
    for i in range(B):
        for j in range(k):
            s = np.random.binomial(n=1,p=p[j])
            x[2*j+s] += 1
    update = ga(x)
    for i in range(0,N-k*B,b):
        for j in range(min(N-k*B-i,b)):
            arm = np.random.choice(k, p=update)
            s = np.random.binomial(n=1,p=p[arm])
            x[2*arm+s] += 1
        update = ga(x)
    for i in range(len(x)):
        x[i] -= 1
    return x

# Calculates RS approximation of posterior probability for a given number of iterations
def rs(iters,x):
    k = len(x)//2
    out = [0 for i in range(k)]
    for i in range(iters):
        sample = list(np.random.beta([x[2*j+1] for j in range(k)],[x[2*j] for j in range(k)]))
        out[sample.index(max(sample))] += 1/iters
    return out

# Runs a TS trial with burn-in using GA approximation with a given number of iterations
def TS_rs(N,p,B,b,iters):
    k = len(p)
    x = [1 for i in range(2*k)]
    for i in range(B):
        for j in range(k):
            s = np.random.binomial(n=1,p=p[j])
            x[2*j+s] += 1
    update = rs(iters,x)
    for i in range(0,N-k*B,b):
        for j in range(min(N-k*B-i,b)):
            arm = np.random.choice(k, p=update)
            s = np.random.binomial(n=1,p=p[arm])
            x[2*arm+s] += 1
        update = rs(iters,x)
    for i in range(len(x)):
        x[i] -= 1
    return x
