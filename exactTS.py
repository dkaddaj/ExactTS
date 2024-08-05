import numpy as np
import pickle
import math
import random

with open ('F_200.txt', 'rb') as file:
    F = pickle.load(file)

def dmult(a,b):
    return [a[i]*b[0][i]+a[i+1]*b[1][i] for i in range(len(b[0]))]

def mult4_0(E,A_0):
    return [[E[i][j]*A_0[0][i][j]+E[i+1][j]*A_0[1][i][j] for j in range(len(E[i]))] for i in range(len(A_0[0]))]

def mult4_1(E,A_1):
    return [dmult(E[i],A_1[i]) for i in range(len(A_1))]

def Dmult(A,B):
    return [list(np.array(mult4_0(A[i],B[0][i]))+np.array(mult4_1(A[i+1],B[1][i]))) for i in range(len(B[0]))]

# Exact probability of rejecting H_0 with Bayesian-based test and summary statistics Markov chain
def test_exact(N,p_0,p_1,c):
    A_0 = {}
    A_1 = {}
    B = {}
    for n in range(N+1):
        B[n] = None
        for m in range(N+1-n):
            A_0[n,m] = None
            A_1[n,m] = None    
    for n in range(N+1):
        for m in range(N+1-n):
            A_0[n,m] = [[[(1-F[n-i][i][m-j][j])*p_0 for j in range(m+1)] for i in range(n+1)],[[(1-F[n-i][i][m-j][j])*(1-p_0) for j in range(m+1)] for i in range(n+1)]]
            A_1[n,m] = [[[F[n-i][i][m-j][j]*p_1 for j in range (m+1)],[F[n-i][i][m-j][j]*(1-p_1) for j in range (m+1)]] for i in range (n+1)]
    calE2 = [[[(F[N-k-i][i][k-j][j]>=c)+(F[N-k-i][i][k-j][j]<=1-c) for j in range(k+1)] for i in range(N-k+1)] for k in range(N+1)]
    for n in range(N):
        B[n] = [[[[[0 for q in range(m+1)] for i in range(n-m+1)]]*2 for m in range(n+1)],[[[[0 for j in range (m+1)]]*2 for i in range (n-m+1)] for m in range(n+1)]]
        for i in range(n+1):
            B[n][0][i] = A_0[n-i,i]
        for i in range(n+1):
            B[n][1][i] = A_1[n-i,i]
    for i in range(N):
        calE2 = Dmult(calE2,B[N-1-i])
    return calE2[0][0][0]

def clipper(prob,lb,ub):
    if prob < lb:
        return lb
    elif prob > ub:
        return ub
    else:
        return prob

def clipped_2test_exact(N,p_0,p_1,c,lb,ub):
    A_0 = {}
    A_1 = {}
    B = {}
    for n in range(N+1):
        B[n] = None
        for m in range(N+1-n):
            A_0[n,m] = None
            A_1[n,m] = None    
    for n in range(N+1):
        for m in range(N+1-n):
            A_0[n,m] = [[[clipper(1-F[n-i][i][m-j][j],lb,ub)*p_0 for j in range(m+1)] for i in range(n+1)],[[clipper(1-F[n-i][i][m-j][j],lb,ub)*(1-p_0) for j in range(m+1)] for i in range(n+1)]]
            A_1[n,m] = [[[clipper(F[n-i][i][m-j][j],lb,ub)*p_1 for j in range (m+1)],[clipper(F[n-i][i][m-j][j],lb,ub)*(1-p_1) for j in range (m+1)]] for i in range (n+1)]
    calE2 = [[[(F[N-k-i][i][k-j][j]>=c)+(F[N-k-i][i][k-j][j]<=1-c) for j in range(k+1)] for i in range(N-k+1)] for k in range(N+1)]
    for n in range(N):
        B[n] = [[[[[0 for q in range(m+1)] for i in range(n-m+1)]]*2 for m in range(n+1)],[[[[0 for j in range (m+1)]]*2 for i in range (n-m+1)] for m in range(n+1)]]
        for i in range(n+1):
            B[n][0][i] = A_0[n-i,i]
        for i in range(n+1):
            B[n][1][i] = A_1[n-i,i]
    for i in range(N):
        calE2 = Dmult(calE2,B[N-1-i])
    return calE2[0][0][0]

def truncated(N,block,p_0,p_1,c,lb,ub):
    A_0 = {}
    A_1 = {}
    B = {}
    for n in range(N+1):
        B[n] = None
        for m in range(N+1-n):
            A_0[n,m] = None
            A_1[n,m] = None    
    for n in range(N+1):
        for m in range(N+1-n):
            A_0[n,m] = [[[clipper(1-F[n-i][i][m-j][j],lb,ub)*p_0 for j in range(m+1)] for i in range(n+1)],[[clipper(1-F[n-i][i][m-j][j],lb,ub)*(1-p_0) for j in range(m+1)] for i in range(n+1)]]
            A_1[n,m] = [[[clipper(F[n-i][i][m-j][j],lb,ub)*p_1 for j in range (m+1)],[clipper(F[n-i][i][m-j][j],lb,ub)*(1-p_1) for j in range (m+1)]] for i in range (n+1)]
    for M in range(block,N,block):
        for n in range(M+1):
            m = M - n
            A_0[n,m] = [[[clipper(1-F[n-i][i][m-j][j],lb,ub)*p_0*(1-(F[n-i][i][m-j][j]>=c)-(F[n-i][i][m-j][j]<=1-c)) for j in range(m+1)] for i in range(n+1)],[[clipper(1-F[n-i][i][m-j][j],lb,ub)*(1-p_0)*(1-(F[n-i][i][m-j][j]>=c)-(F[n-i][i][m-j][j]<=1-c)) for j in range(m+1)] for i in range(n+1)]]
            A_1[n,m] = [[[clipper(F[n-i][i][m-j][j],lb,ub)*p_1*(1-(F[n-i][i][m-j][j]>=c)-(F[n-i][i][m-j][j]<=1-c)) for j in range (m+1)],[clipper(F[n-i][i][m-j][j],lb,ub)*(1-p_1)*(1-(F[n-i][i][m-j][j]>=c)-(F[n-i][i][m-j][j]<=1-c)) for j in range (m+1)]] for i in range (n+1)]

    calE2 = [[[(F[N-k-i][i][k-j][j]>=c)+(F[N-k-i][i][k-j][j]<=1-c) for j in range(k+1)] for i in range(N-k+1)] for k in range(N+1)]
    for n in range(N):
        B[n] = [[[[[0 for q in range(m+1)] for i in range(n-m+1)]]*2 for m in range(n+1)],[[[[0 for j in range (m+1)]]*2 for i in range (n-m+1)] for m in range(n+1)]]
        for i in range(n+1):
            B[n][0][i] = A_0[n-i,i]
        for i in range(n+1):
            B[n][1][i] = A_1[n-i,i]
    for i in range(N):
        calE2 = Dmult(calE2,B[N-1-i])
    return calE2[0][0][0]

# Exact probability of rejecting H_0 with Bayesian blocked RA design with early stopping 
def blocked_test_exact(N,block,p_0,p_1,c,lb,ub):
    out = clipped_2test_exact(block,p_0,p_1,c,lb,ub)
    for i in range(2*block,N+1,block):
        out += truncated(i,block,p_0,p_1,c,lb,ub)
    return out

# Exact expected trail size for Bayesian blocked RA design with early stopping 
def blocked_trial_exact(N,block,p_0,p_1,c,lb,ub):
    a = clipped_2test_exact(block,p_0,p_1,c,lb,ub)
    out = a*block
    prob = a
    for i in range(2*block,N+1,block):
        a = truncated(i,block,p_0,p_1,c,lb,ub)
        out += a*i
        prob += a
    return out + N*(1-prob)

def TS_sim(N,p_0,p_1):
    outcomes = [0,0,0,0]
    for n in range(N):
        x = np.random.beta(outcomes[0]+1,outcomes[1]+1)
        y = np.random.beta(outcomes[2]+1,outcomes[3]+1)
        s = random.random()
        if y < x:
            if s < p_0:
                outcomes[0] += 1
            else:
                outcomes[1] += 1
        else:
            if s < p_1:
                outcomes[2] += 1
            else:
                outcomes[3] += 1
    return outcomes

# Simulated probability of rejecting H_0 with Bayesian-based test and summary statistics Markov chain
def test_sim(N,p_0,p_1,c,iters):
    sim_out = 0
    for i in range(iters):
        outcomes = TS_sim(N,p_0,p_1)
        sim_out += ((F[outcomes[0]][outcomes[1]][outcomes[2]][outcomes[3]]>=c)+(F[outcomes[0]][outcomes[1]][outcomes[2]][outcomes[3]]<=1-c))/iters
    return sim_out

def blocked_TS_sim(N,block,p_0,p_1,c,lb,ub):
    outcomes = [0,0,0,0]
    for b in range(N//block):
        for n in range(block):
            if F[outcomes[0]][outcomes[1]][outcomes[2]][outcomes[3]]>=ub:
                x = random.random()
                s = random.random()
                if x > ub:
                    if s < p_0:
                        outcomes[0] += 1
                    else:
                        outcomes[1] += 1
                else:
                    if s < p_1:
                        outcomes[2] += 1
                    else:
                        outcomes[3] += 1
            elif F[outcomes[0]][outcomes[1]][outcomes[2]][outcomes[3]]<=lb:
                x = random.random()
                s = random.random()
                if x > lb:
                    if s < p_0:
                        outcomes[0] += 1
                    else:
                        outcomes[1] += 1
                else:
                    if s < p_1:
                        outcomes[2] += 1
                    else:
                        outcomes[3] += 1
            else:
                x = np.random.beta(outcomes[0]+1,outcomes[1]+1)
                y = np.random.beta(outcomes[2]+1,outcomes[3]+1)
                s = random.random()
                if y < x:
                    if s < p_0:
                        outcomes[0] += 1
                    else:
                        outcomes[1] += 1
                else:
                    if s < p_1:
                        outcomes[2] += 1
                    else:
                        outcomes[3] += 1
        if (F[outcomes[0]][outcomes[1]][outcomes[2]][outcomes[3]]>=c)+(F[outcomes[0]][outcomes[1]][outcomes[2]][outcomes[3]]<=1-c) == 1:
            return 1
    return 0

def blocked_trialTS_sim(N,block,p_0,p_1,c,lb,ub):
    outcomes = [0,0,0,0]
    for b in range(N//block):
        for n in range(block):
            if F[outcomes[0]][outcomes[1]][outcomes[2]][outcomes[3]]>=ub:
                x = random.random()
                s = random.random()
                if x > ub:
                    if s < p_0:
                        outcomes[0] += 1
                    else:
                        outcomes[1] += 1
                else:
                    if s < p_1:
                        outcomes[2] += 1
                    else:
                        outcomes[3] += 1
            elif F[outcomes[0]][outcomes[1]][outcomes[2]][outcomes[3]]<=lb:
                x = random.random()
                s = random.random()
                if x > lb:
                    if s < p_0:
                        outcomes[0] += 1
                    else:
                        outcomes[1] += 1
                else:
                    if s < p_1:
                        outcomes[2] += 1
                    else:
                        outcomes[3] += 1
            else:
                x = np.random.beta(outcomes[0]+1,outcomes[1]+1)
                y = np.random.beta(outcomes[2]+1,outcomes[3]+1)
                s = random.random()
                if y < x:
                    if s < p_0:
                        outcomes[0] += 1
                    else:
                        outcomes[1] += 1
                else:
                    if s < p_1:
                        outcomes[2] += 1
                    else:
                        outcomes[3] += 1
        if (F[outcomes[0]][outcomes[1]][outcomes[2]][outcomes[3]]>=c)+(F[outcomes[0]][outcomes[1]][outcomes[2]][outcomes[3]]<=1-c) == 1:
            return sum(outcomes)
    return N

# Simulated probability of rejecting H_0 with Bayesian blocked RA design with early stopping 
def blocked_test_sim(N,block,p_0,p_1,c,lb,ub,iters):
    sim_out = 0
    for i in range(iters):
        sim_out += blocked_TS_sim(N,block,p_0,p_1,c,lb,ub)/iters
    return sim_out

def blocked_trial_sim(N,block,p_0,p_1,c,lb,ub,iters):
    sim_out = 0
    for i in range(iters):
        sim_out += blocked_trialTS_sim(N,block,p_0,p_1,c,lb,ub)/iters
    return sim_out

# Regula falsi algorithm to find to 5 d.p. exact c that controls type I error to aim in Bayesian-based test and summary statistics Markov chain with p_0=p_1=p
def c_finder(N,p,aim):
    top = [0,1]
    bottom = [1,0]
    while abs(bottom[0]-top[0])>0.00001:
        x = top[0]+(top[1]-aim)*(bottom[0]-top[0])/(top[1]-bottom[1])
        temp = [x,test_exact(N,p,p,x)]
        if temp[1]>aim:
            top = temp
        else:
            bottom = temp
    target = math.ceil(top[0]*100000)/100000
    if top[0]< target-0.000005:
        x = target-0.000005
        temp = [x,test_exact(N,p,p,x)]
        if temp[1]<=aim:
            target -= 0.00001
    else:
        x = target+0.000005
        temp = [x,test_exact(N,p,p,x)]
        if temp[1]>aim:
            target += 0.00001
    return target

# Regula falsi algorithm to find c to 5 d.p. based on simulations, that controls type I error to aim in Bayesian-based test and summary statistics Markov chain with p_0=p_1=p
def c_finder_sim(N,p,aim,iters):
    top = [0,1]
    bottom = [1,0]
    while abs(bottom[0]-top[0])>0.00001:
        x = top[0]+(top[1]-aim)*(bottom[0]-top[0])/(top[1]-bottom[1])
        temp = [x,test_sim(N,p,p,x,iters)]
        if temp[1]>aim:
            top = temp
        else:
            bottom = temp
    target = math.ceil(top[0]*100000)/100000
    if top[0]< target-0.000005:
        x = target-0.000005
        temp = [x,test_sim(N,p,p,x,iters)]
        if temp[1]<=aim:
            target -= 0.00001
    else:
        x = target+0.000005
        temp = [x,test_sim(N,p,p,x,iters)]
        if temp[1]>aim:
            target += 0.00001
    return target

def arm_exact(N,p_0,p_1):
    A_0 = {}
    A_1 = {}
    B = {}
    for n in range(N+1):
        B[n] = None
        for m in range(N+1-n):
            A_0[n,m] = None
            A_1[n,m] = None    
    for n in range(N+1):
        for m in range(N+1-n):
            A_0[n,m] = [[[(1-F[n-i][i][m-j][j])*p_0 for j in range(m+1)] for i in range(n+1)],[[(1-F[n-i][i][m-j][j])*(1-p_0) for j in range(m+1)] for i in range(n+1)]]
            A_1[n,m] = [[[F[n-i][i][m-j][j]*p_1 for j in range (m+1)],[F[n-i][i][m-j][j]*(1-p_1) for j in range (m+1)]] for i in range (n+1)]
    calE2 = [[[k for j in range(k+1)] for i in range(N-k+1)] for k in range(N+1)]
    for n in range(N):
        B[n] = [[[[[0 for q in range(m+1)] for i in range(n-m+1)]]*2 for m in range(n+1)],[[[[0 for j in range (m+1)]]*2 for i in range (n-m+1)] for m in range(n+1)]]
        for i in range(n+1):
            B[n][0][i] = A_0[n-i,i]
        for i in range(n+1):
            B[n][1][i] = A_1[n-i,i]
    for i in range(N):
        calE2 = Dmult(calE2,B[N-1-i])
    return calE2[0][0][0]

def arm_sim(N,p_0,p_1,iters):
    sim_out = 0
    for i in range(iters):
        outcomes = TS_sim(N,p_0,p_1)
        sim_out += (outcomes[2]+outcomes[3])/iters
    return sim_out

def blocked_c_finder(N,block,p,lb,ub,aim):
    top = [0,1]
    bottom = [1,0]
    while abs(bottom[0]-top[0])>0.00001:
        x = top[0]+(top[1]-aim)*(bottom[0]-top[0])/(top[1]-bottom[1])
        temp = [x,blocked_test_exact(N,block,p,p,x,lb,ub)]
        if temp[1]>aim:
            top = temp
        else:
            bottom = temp
    target = math.ceil(top[0]*100000)/100000
    if top[0]< target-0.000005:
        x = target-0.000005
        temp = [x,blocked_test_exact(N,block,p,p,x,lb,ub)]
        if temp[1]<=aim:
            target -= 0.00001
    else:
        x = target+0.000005
        temp = [x,blocked_test_exact(N,block,p,p,x,lb,ub)]
        if temp[1]>aim:
            target += 0.00001
    return target

def blocked_c_finder_sim(N,block,p,lb,ub,aim,iters):
    top = [0,1]
    bottom = [1,0]
    while abs(bottom[0]-top[0])>0.00001:
        x = top[0]+(top[1]-aim)*(bottom[0]-top[0])/(top[1]-bottom[1])
        temp = [x,blocked_test_sim(N,block,p,p,x,lb,ub,iters)]
        if temp[1]>aim:
            top = temp
        else:
            bottom = temp
    target = math.ceil(top[0]*100000)/100000
    if top[0]< target-0.000005:
        x = target-0.000005
        temp = [x,blocked_test_sim(N,block,p,p,x,lb,ub,iters)]
        if temp[1]<=aim:
            target -= 0.00001
    else:
        x = target+0.000005
        temp = [x,blocked_test_sim(N,block,p,p,x,lb,ub,iters)]
        if temp[1]>aim:
            target += 0.00001
    return target
