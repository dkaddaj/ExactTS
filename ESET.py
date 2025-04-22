# Code to generate Tables 1 and 2 corresponding to the ESET trial

import scipy.special as sc
from scipy.stats import multivariate_normal
from scipy.stats import multinomial
import numpy as np
import itertools
import math
import time

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
            return (1-p[0]-p[1]>p[1] and 1-p[0]-p[1]>p[1])
        elif prob[0] > 0.975:
            return (p[0]>1-p[0]-p[1] and p[0]>p[1])
        elif prob[1] > 0.975:
            return (p[1]>1-p[0]-p[1] and p[1]>p[0])
        if sc.betainc(temp[1]+1,temp[0]+1,0.25)>0.95:
            dropped.add(0)
        if sc.betainc(temp[3]+1,temp[2]+1,0.25)>0.95:
            dropped.add(1)
        if sc.betainc(temp[5]+1,temp[4]+1,0.25)>0.95:
            dropped.add(2)
        if len(dropped) == 3:
            return 0
    rtemp = [0,0,0,0,0,0]
    routcomes = [temp[1],temp[0],temp[3],temp[2],temp[5],temp[4]]
    rprob = [1/3,1/3]
    rhist = [s_post_2([0,0,1,1]),s_post_2([0,0,1,1]),s_post_2([0,0,1,1])]
    for i in range(6):
        for j in range(routcomes[i]):
            rtemp[i] += 1
            [rprob,rhist] = r_post_3(i,rtemp,rprob,rhist)
    if 1-prob[0]-prob[1] > 0.975:
        return (1-p[0]-p[1]>p[1] and 1-p[0]-p[1]>p[1])
    elif prob[0] > 0.975:
        return (p[0]>1-p[0]-p[1] and p[0]>p[1])
    elif prob[1] > 0.975:
        return (p[1]>1-p[0]-p[1] and p[1]>p[0])
    elif 1-rprob[0]-rprob[1] > 0.975:
        return (1-p[0]-p[1]<p[1] and 1-p[0]-p[1]<p[1])
    elif rprob[0] > 0.975:
        return (p[0]<1-p[0]-p[1] and p[0]<p[1])
    elif rprob[1] > 0.975:
        return (p[1]<1-p[0]-p[1] and p[1]<p[0])
    else:
        return (p[0]==p[1]==p[2])

# Calculates GA posterior
def ga(x):
    mu = []
    sigma = []
    for i in range(3):
        mu.append((x[2*i]+1)/(x[2*i]+x[2*i+1]+2))
        sigma.append(((x[2*i]+1)*(x[2*i+1]+1))/((x[2*i]+x[2*i+1]+3)*((x[2*i]+x[2*i+1]+2)**2)))
    return [multivariate_normal.cdf([0,0],mean=[mu[1]-mu[0],mu[1]-mu[2]],cov=[[sigma[0]+sigma[1],sigma[1]],[sigma[1],sigma[1]+sigma[2]]]),multivariate_normal.cdf([0,0],mean=[mu[2]-mu[0],mu[2]-mu[1]],cov=[[sigma[0]+sigma[2],sigma[2]],[sigma[2],sigma[1]+sigma[2]]])]

# Uses GA calculation to run an ESET trial with 720 patients, response probabilites p, burn-in B per arm, and frequency of subsequent randomisation b
def ESET_ga(p,B,b):
    N = 720
    temp = [0,0,0,0,0,0]
    dropped = set()
    temp[1] += np.random.binomial(n=B, p=p[0])
    temp[0] = B - temp[1]
    temp[3] += np.random.binomial(n=B, p=p[1])
    temp[2] = B - temp[3]
    temp[5] += np.random.binomial(n=B, p=p[2])
    temp[4] = B - temp[5]
    intervals = [b for i in range((N-3*B)//b)]
    if (N-3*B) % b != 0:
        intervals.append((N-3*B)%b)
    prob = ga(temp)
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
        prob = ga(temp)
        if 1-prob[0]-prob[1] > 0.975:
            return (1-p[0]-p[1]>p[1] and 1-p[0]-p[1]>p[1])
        elif prob[0] > 0.975:
            return (p[0]>1-p[0]-p[1] and p[0]>p[1])
        elif prob[1] > 0.975:
            return (p[1]>1-p[0]-p[1] and p[1]>p[0])
        if sc.betainc(temp[1]+1,temp[0]+1,0.25)>0.95:
            dropped.add(0)
        if sc.betainc(temp[3]+1,temp[2]+1,0.25)>0.95:
            dropped.add(1)
        if sc.betainc(temp[5]+1,temp[4]+1,0.25)>0.95:
            dropped.add(2)
        if len(dropped) == 3:
            return 0
    rtemp = [temp[1],temp[0],temp[3],temp[2],temp[5],temp[4]]
    rprob = ga(rtemp)
    if 1-prob[0]-prob[1] > 0.975:
        return (1-p[0]-p[1]>p[1] and 1-p[0]-p[1]>p[1])
    elif prob[0] > 0.975:
        return (p[0]>1-p[0]-p[1] and p[0]>p[1])
    elif prob[1] > 0.975:
        return (p[1]>1-p[0]-p[1] and p[1]>p[0])
    elif 1-rprob[0]-rprob[1] > 0.975:
        return (1-p[0]-p[1]<p[1] and 1-p[0]-p[1]<p[1])
    elif rprob[0] > 0.975:
        return (p[0]<1-p[0]-p[1] and p[0]<p[1])
    elif rprob[1] > 0.975:
        return (p[1]<1-p[0]-p[1] and p[1]<p[0])
    else:
        return (p[0]==p[1]==p[2])

# Calculates RS posterior with 10,000 iterations
def rs(x):
    out = [0,0]
    for i in range(10000):
        sample = list(np.random.beta([x[2*j+1] for j in range(3)],[x[2*j] for j in range(3)]))
        out[sample.index(max(sample))] += 1/iters
    return [out[1],out[2]]

# Uses RS with 10,000 iterations to run an ESET trial with 720 patients, response probabilites p, burn-in B per arm, and frequency of subsequent randomisation b
def ESET_rs(p,B,b):
    N = 720
    temp = [0,0,0,0,0,0]
    dropped = set()
    temp[1] += np.random.binomial(n=B, p=p[0])
    temp[0] = B - temp[1]
    temp[3] += np.random.binomial(n=B, p=p[1])
    temp[2] = B - temp[3]
    temp[5] += np.random.binomial(n=B, p=p[2])
    temp[4] = B - temp[5]
    intervals = [b for i in range((N-3*B)//b)]
    if (N-3*B) % b != 0:
        intervals.append((N-3*B)%b)
    prob = rs(temp)
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
        prob = rs(temp)
        if 1-prob[0]-prob[1] > 0.975:
            return (1-p[0]-p[1]>p[1] and 1-p[0]-p[1]>p[1])
        elif prob[0] > 0.975:
            return (p[0]>1-p[0]-p[1] and p[0]>p[1])
        elif prob[1] > 0.975:
            return (p[1]>1-p[0]-p[1] and p[1]>p[0])
        if sc.betainc(temp[1]+1,temp[0]+1,0.25)>0.95:
            dropped.add(0)
        if sc.betainc(temp[3]+1,temp[2]+1,0.25)>0.95:
            dropped.add(1)
        if sc.betainc(temp[5]+1,temp[4]+1,0.25)>0.95:
            dropped.add(2)
        if len(dropped) == 3:
            return 0
    rtemp = [temp[1],temp[0],temp[3],temp[2],temp[5],temp[4]]
    rprob = rs(rtemp)
    if 1-prob[0]-prob[1] > 0.975:
        return (1-p[0]-p[1]>p[1] and 1-p[0]-p[1]>p[1])
    elif prob[0] > 0.975:
        return (p[0]>1-p[0]-p[1] and p[0]>p[1])
    elif prob[1] > 0.975:
        return (p[1]>1-p[0]-p[1] and p[1]>p[0])
    elif 1-rprob[0]-rprob[1] > 0.975:
        return (1-p[0]-p[1]<p[1] and 1-p[0]-p[1]<p[1])
    elif rprob[0] > 0.975:
        return (p[0]<1-p[0]-p[1] and p[0]<p[1])
    elif rprob[1] > 0.975:
        return (p[1]<1-p[0]-p[1] and p[1]<p[0])
    else:
        return (p[0]==p[1]==p[2])

# Generate Table 1
def Tab1():
    print('Exact computation:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            start = time.time()
            for i in range(10000):
                out += ESET([0.5,0.5,0.5],B,b)
            end = time.time()
            print('Time taken for B =',B,'and b =',b,'is:',end-start)
    print('GA:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            start = time.time()
            for i in range(10000):
                out += ESET_ga([0.5,0.5,0.5],B,b)
            end = time.time()
            print('Time taken for B =',B,'and b =',b,'is:',end-start)
    print('RS:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            start = time.time()
            for i in range(10000):
                out += ESET_rs([0.5,0.5,0.5],B,b)
            end = time.time()
            print('Time taken for B =',B,'and b =',b,'is:',end-start)

# Generate Type I error rate data in Table 2
def Tab2a():
    print('Exact computation:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            for i in range(10000):
                out += ESET([0.5,0.5,0.5],B,b)
            print('The type I error rate for B =',B,'and b =',b,'is:',out)
    print('GA:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            for i in range(10000):
                out += ESET_ga([0.5,0.5,0.5],B,b)
            print('The type I error rate for B =',B,'and b =',b,'is:',out)
    print('RS:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            for i in range(10000):
                out += ESET_rs([0.5,0.5,0.5],B,b)
            print('The type I error rate for B =',B,'and b =',b,'is:',out)

# Generate power data for p = (0.5,0.5,0.65) in Table 2
def Tab2b():
    print('Exact computation:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            for i in range(10000):
                out += ESET([0.5,0.5,0.65],B,b)
            print('The power for B =',B,'and b =',b,'is:',out)
    print('GA:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            for i in range(10000):
                out += ESET_ga([0.5,0.5,0.65],B,b)
            print('The power for B =',B,'and b =',b,'is:',out)
    print('RS:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            for i in range(10000):
                out += ESET_rs([0.5,0.5,0.65],B,b)
            print('The power for B =',B,'and b =',b,'is:',out)

# Generate power data for p = (0.5,0.65,0.65) in Table 2
def Tab2c():
    print('Exact computation:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            for i in range(10000):
                out += ESET([0.5,0.65,0.65],B,b)
            print('The power for B =',B,'and b =',b,'is:',out)
    print('GA:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            for i in range(10000):
                out += ESET_ga([0.5,0.65,0.65],B,b)
            print('The power for B =',B,'and b =',b,'is:',out)
    print('RS:')
    for B in [100,50,0]:
        for b in [100,20,5,1]:
            out = 0
            for i in range(10000):
                out += ESET_rs([0.5,0.65,0.65],B,b)
            print('The power for B =',B,'and b =',b,'is:',out)
