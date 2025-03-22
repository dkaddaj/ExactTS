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
def TS_exact(N,p,burn_in):
    k = len(p)
    x = [1 for i in range(2*k)]
    hist = load_hist(k)
    for i in range(burn_in):
        for j in range(k):
            s = np.random.binomial(n=1,p=p[j])
            x,hist = update_full(x,hist,j,s)
    update = [hist[(i,)] for i in range(k)]
    for i in range(N-k*burn_in):
        j = np.random.choice(k, p=update)
        s = np.random.binomial(n=1,p=p[j])
        x,hist = update_full(x,hist,j,s)
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
def TS_ga(N,p,burn_in):
    k = len(p)
    x = [1 for i in range(2*k)]
    for i in range(burn_in):
        for j in range(k):
            s = np.random.binomial(n=1,p=p[j])
            x[2*j+s] += 1
    update = ga(x)
    for i in range(N-k*burn_in):
        j = np.random.choice(k, p=update)
        s = np.random.binomial(n=1,p=p[j])
        x[2*j+s] += 1
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
def TS_rs(N,p,burn_in,iters):
    k = len(p)
    x = [1 for i in range(2*k)]
    for i in range(burn_in):
        for j in range(k):
            s = np.random.binomial(n=1,p=p[j])
            x[2*j+s] += 1
    update = rs(iters,x)
    for i in range(N-k*burn_in):
        j = np.random.choice(k, p=update)
        s = np.random.binomial(n=1,p=p[j])
        x[2*j+s] += 1
        update = rs(iters,x)
    for i in range(len(x)):
        x[i] -= 1
    return x

# Auxiliary function which computes posterior probability for two treatments
def s_post_2(state):
    [i,j,k,l] = state
    smallest = min(i,j,k,l)
    if smallest == i:
        return sum([sc.beta(k+m+1,j+l+2)/((j+m+1)*sc.beta(m+1,j+1)*sc.beta(k+1,l+1)) for m in range(i+1)])
    elif smallest == j:
        return 1-sum([sc.beta(l+m+1,i+k+2)/((i+m+1)*sc.beta(m+1,i+1)*sc.beta(k+1,l+1)) for m in range(j+1)])
    elif smallest == k:
        return 1-sum([sc.beta(i+m+1,j+l+2)/((l+m+1)*sc.beta(m+1,l+1)*sc.beta(i+1,j+1)) for m in range(k+1)])
    else:
        return sum([sc.beta(j+m+1,i+k+2)/((k+m+1)*sc.beta(m+1,k+1)*sc.beta(i+1,j+1)) for m in range(l+1)])

# Auxiliary function which computes an increment in the posterior probability for two treatments
def r_post_2(incr,state,prob):
    s = state.copy()
    s[incr] -= 1
    [i,j,k,l] = s
    temp = sc.beta(i+k+2,j+l+2)/(sc.beta(i+1,j+1)*sc.beta(k+1,l+1))
    if incr == 0 or incr == 3:
        return prob + temp/(s[incr]+1)
    else:
        return prob - temp/(s[incr]+1)

# Auxiliary function which computes an increment in the posterior probability for three treatments
def r_post_3(incr,state,prob,hist):
    s = state.copy()
    s[incr] -= 1
    [a,b,c,d,e,f] = s
    if incr == 0:
        return [[prob[0]+hist[2]*sc.beta(a+c+2,b+d+2)/((a+1)*sc.beta(a+1,b+1)*sc.beta(c+1,d+1)),prob[1]+hist[1]*sc.beta(a+e+2,b+f+2)/((a+1)*sc.beta(a+1,b+1)*sc.beta(e+1,f+1))],[r_post_2(0,[a+1,b,c+e+1,d+f+1],hist[0]),r_post_2(2,[c,d,a+e+2,b+f+1],hist[1]),r_post_2(2,[e,f,a+c+2,b+d+1],hist[2])]]
    elif incr == 1:
        return [[prob[0]-hist[2]*sc.beta(a+c+2,b+d+2)/((b+1)*sc.beta(a+1,b+1)*sc.beta(c+1,d+1)),prob[1]-hist[1]*sc.beta(a+e+2,b+f+2)/((b+1)*sc.beta(a+1,b+1)*sc.beta(e+1,f+1))],[r_post_2(1,[a,b+1,c+e+1,d+f+1],hist[0]),r_post_2(3,[c,d,a+e+1,b+f+2],hist[1]),r_post_2(3,[e,f,a+c+1,b+d+2],hist[2])]]
    elif incr == 2:
        return [[prob[0]-hist[2]*sc.beta(a+c+2,b+d+2)/((c+1)*sc.beta(a+1,b+1)*sc.beta(c+1,d+1))-hist[0]*sc.beta(c+e+2,d+f+2)/((c+1)*sc.beta(c+1,d+1)*sc.beta(e+1,f+1)),prob[1]+hist[0]*sc.beta(c+e+2,d+f+2)/((c+1)*sc.beta(c+1,d+1)*sc.beta(e+1,f+1))],[r_post_2(2,[a,b,c+e+2,d+f+1],hist[0]),r_post_2(0,[c+1,d,a+e+1,b+f+1],hist[1]),r_post_2(2,[e,f,a+c+2,b+d+1],hist[2])]]
    elif incr == 3:
        return [[prob[0]+hist[2]*sc.beta(a+c+2,b+d+2)/((d+1)*sc.beta(a+1,b+1)*sc.beta(c+1,d+1))+hist[0]*sc.beta(c+e+2,d+f+2)/((d+1)*sc.beta(c+1,d+1)*sc.beta(e+1,f+1)),prob[1]-hist[0]*sc.beta(c+e+2,d+f+2)/((d+1)*sc.beta(c+1,d+1)*sc.beta(e+1,f+1))],[r_post_2(3,[a,b,c+e+1,d+f+2],hist[0]),r_post_2(1,[c,d+1,a+e+1,b+f+1],hist[1]),r_post_2(3,[e,f,a+c+1,b+d+2],hist[2])]]
    elif incr == 4:
        return [[prob[0]+hist[0]*sc.beta(c+e+2,d+f+2)/((e+1)*sc.beta(c+1,d+1)*sc.beta(e+1,f+1)),prob[1]-hist[1]*sc.beta(a+e+2,b+f+2)/((e+1)*sc.beta(a+1,b+1)*sc.beta(e+1,f+1))-hist[0]*sc.beta(c+e+2,d+f+2)/((e+1)*sc.beta(c+1,d+1)*sc.beta(e+1,f+1))],[r_post_2(2,[a,b,c+e+2,d+f+1],hist[0]),r_post_2(2,[c,d,a+e+2,b+f+1],hist[1]),r_post_2(0,[e+1,f,a+c+1,b+d+1],hist[2])]]
    else:
        return [[prob[0]-hist[0]*sc.beta(c+e+2,d+f+2)/((f+1)*sc.beta(c+1,d+1)*sc.beta(e+1,f+1)),prob[1]+hist[1]*sc.beta(a+e+2,b+f+2)/((f+1)*sc.beta(a+1,b+1)*sc.beta(e+1,f+1))+hist[0]*sc.beta(c+e+2,d+f+2)/((f+1)*sc.beta(c+1,d+1)*sc.beta(e+1,f+1))],[r_post_2(3,[a,b,c+e+1,d+f+2],hist[0]),r_post_2(3,[c,d,a+e+1,b+f+2],hist[1]),r_post_2(1,[e,f+1,a+c+1,b+d+1],hist[2])]]

# Uses exact calculation to run an ESET trial with 720 patients, response probabilites p, burn-in B per arm, and frequency of subsequent randomisation b
# Use the functions ga and rs appropriately if approximations for randomisation or testing are desired
def ESET(p,B,b):
    N = 720
    outcomes = [0,0,0,0,0,0]
    dropped = set()
    outcomes[1] += np.random.binomial(n=B, p=p[0])
    outcomes[0] = B - outcomes[1]
    outcomes[3] += np.random.binomial(n=B, p=p[1])
    outcomes[2] = B - outcomes[3]
    outcomes[5] += np.random.binomial(n=B, p=p[2])
    outcomes[4] = B - outcomes[5]
    intervals = [b for i in range((N-3*B)//b)]
    if (N-3*B) % b != 0:
        intervals.append((N-3*B)%b)
    prob = [1/3,1/3]
    hist = [s_post_2([0,0,1,1]),s_post_2([0,0,1,1]),s_post_2([0,0,1,1])]
    temp = [0,0,0,0,0,0]
    for i in range(6):
        for j in range(outcomes[i]):
            temp[i] += 1
            [prob,hist] = r_post_3(i,temp,prob,hist)
    for num in intervals:
        prob = [1-prob[0]-prob[1],prob[0],prob[1]]
        prob = [max(0,min(1,x)) for x in prob]
        prob = [prob[1],prob[2]]
        randomise = [math.sqrt(max(0,1-prob[0]-prob[1])*(temp[0]+1)*(temp[1]+1)/((temp[0]+temp[1]+1)*(temp[0]+temp[1]+2)*(temp[0]+temp[1]+2)*(temp[0]+temp[1]+3))),math.sqrt(prob[0]*(temp[2]+1)*(temp[3]+1)/((temp[2]+temp[3]+1)*(temp[2]+temp[3]+2)*(temp[2]+temp[3]+2)*(temp[2]+temp[3]+3))),math.sqrt(prob[1]*(temp[4]+1)*(temp[5]+1)/((temp[4]+temp[5]+1)*(temp[4]+temp[5]+2)*(temp[4]+temp[5]+2)*(temp[4]+temp[5]+3)))]
        for d in dropped:
            randomise[d] = 0
        total = sum(randomise)
        randomise = [x/total for x in randomise]
        for i in range(len(randomise)):
            if randomise[i] < 0.05:
                randomise[i] = 0
            total = sum(randomise)
            randomise = [x/total for x in randomise]
        allocation = multinomial.rvs(num,randomise)
        step = [0,0,0,0,0,0]
        for j in range(3):
            step[2*j] = np.random.binomial(n=allocation[j],p=1-p[j])
            step[2*j+1] = allocation[j]-step[2*j]
        for i in range(6):
            for j in range(step[i]):
                temp[i] += 1
                [prob,hist] = r_post_3(i,temp,prob,hist)
        if 1-prob[0]-prob[1] > 0.975:
            print('Trial stopped early in favour of treatment 0')
            return temp
        elif prob[0] > 0.975:
            print('Trial stopped early in favour of treatment 1')
            return temp
        elif prob[1] > 0.975:
            print('Trial stopped early in favour of treatment 2')
            return temp
        if sc.betainc(temp[1]+1,temp[0]+1,0.25)>0.95:
            if 0 not in dropped:
                print('Treatment 0 has been dropped')
            dropped.add(0)
        if sc.betainc(temp[3]+1,temp[2]+1,0.25)>0.95:
            if 1 not in dropped:
                print('Treatment 1 has been dropped')
            dropped.add(1)
        if sc.betainc(temp[5]+1,temp[4]+1,0.25)>0.95:
            if 2 not in dropped:
                print('Treatment 2 has been dropped')
            dropped.add(2)
        if len(dropped) == 3:
            print('Trial stopped early as all treatments were dropped')
            return temp
    return temp
