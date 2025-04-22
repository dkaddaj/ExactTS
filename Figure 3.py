# Code to generate Figure 3

import pickle
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import griddata

with open ('F_200.txt', 'rb') as file:
    F = pickle.load(file)
with open ('w_200.txt', 'rb') as file:
    w = pickle.load(file)
with open ('w_ga_200.txt', 'rb') as file:
    w_ga = pickle.load(file)

def test_exact(N,p_0,p_1,c):
    if p_0 == p_1:
        return sum([sum([sum([w[i][j][k][N-i-j-k]*((F[i][j][k][N-i-j-k]>=c)+(F[i][j][k][N-i-j-k]<=1-c))*(p_0**i)*((1-p_0)**j)*(p_1**k)*((1-p_1)**(N-i-j-k)) for k in range(N+1-i-j)]) for j in range(N+1-i)]) for i in range(N+1)])/2
    elif p_0 < p_1:
        return sum([sum([sum([w[i][j][k][N-i-j-k]*((1-F[i][j][k][N-i-j-k]<=1-c))*(p_0**i)*((1-p_0)**j)*(p_1**k)*((1-p_1)**(N-i-j-k)) for k in range(N+1-i-j)]) for j in range(N+1-i)]) for i in range(N+1)])
    else:
        return sum([sum([sum([w[i][j][k][N-i-j-k]*((1-F[i][j][k][N-i-j-k]>=c))*(p_0**i)*((1-p_0)**j)*(p_1**k)*((1-p_1)**(N-i-j-k)) for k in range(N+1-i-j)]) for j in range(N+1-i)]) for i in range(N+1)])

def gest_exact(N,p_0,p_1,c):
    if p_0 == p_1:
        return sum([sum([sum([w_ga[i][j][k][N-i-j-k]*((F[i][j][k][N-i-j-k]>=c)+(F[i][j][k][N-i-j-k]<=1-c))*(p_0**i)*((1-p_0)**j)*(p_1**k)*((1-p_1)**(N-i-j-k)) for k in range(N+1-i-j)]) for j in range(N+1-i)]) for i in range(N+1)])/2
    elif p_0 < p_1:
        return sum([sum([sum([w_ga[i][j][k][N-i-j-k]*((1-F[i][j][k][N-i-j-k]<=1-c))*(p_0**i)*((1-p_0)**j)*(p_1**k)*((1-p_1)**(N-i-j-k)) for k in range(N+1-i-j)]) for j in range(N+1-i)]) for i in range(N+1)])
    else:
        return sum([sum([sum([w_ga[i][j][k][N-i-j-k]*((1-F[i][j][k][N-i-j-k]>=c))*(p_0**i)*((1-p_0)**j)*(p_1**k)*((1-p_1)**(N-i-j-k)) for k in range(N+1-i-j)]) for j in range(N+1-i)]) for i in range(N+1)])

def tRS(N,p_0,p_1,c):
    out = 0
    for count in range(100000):
        temp = TS(N,p_0,p_1)
        i = temp[0]
        j = temp[1]
        k = temp[2]
        l = temp[3]
        q = np.random.binomial(n=10000, p=F[i][j][k][l])/10000
        out += (q>=c or 1-q>=c)/100000
    return out

P = [i/50 for i in range(51)]
UX = [test_exact(N,p,p,0.99753) for p in P]
gUXt = [gest_exact(N,p,p,0.99753) for p in P]
rUXt = [tRS(N,p,p,0.99753) for p in P]
PP6 = [test_exact(N,p,p,0.9894462806660532) for p in P]
gPP6t = [gest_exact(N,p,p,0.9894462806660532) for p in P]
rPP6t = [tRS(N,p,p,0.9894462806660532) for p in P]

plt.plot(P,PP6)
plt.plot(P,gPP6t)
plt.plot(P,UX)
plt.plot(P,gUXt)
plt.legend(['PP(0.6)','PP(0.6)_GA','UX','UX_GA'])
plt.xlabel('p')
plt.ylabel('Type I error')
plt.savefig('3a.png')
plt.close()

plt.plot(P,[PP6[i]-rPP6t[i] for i in range(51)])
plt.plot(P,[UX[i]-rUXt[i] for i in range(51)])
plt.legend(['PP(0.6)','UX'])
plt.xlabel('p')
plt.ylabel('Difference in type I error')
plt.savefig('3b.png')
plt.close()

uprob1t = [gest_exact(200,X[i],Y[i],0.99753) for i in range(len(X))]
uprob2t = [tRS(200,X[i],Y[i],0.99753) for i in range(len(X))]
prob1t = [gest_exact(200,X[i],Y[i],0.9894462806660532) for i in range(len(X))]
prob2t = [tRS(200,X[i],Y[i],0.9894462806660532) for i in range(len(X))]

tpowUX_ga = np.array([uprob1t[i]-uprob0[i] for i in range(len(X))])
tpowUX_rs = np.array([uprob2t[i]-uprob0[i] for i in range(len(X))])
tpowPP_ga = np.array([prob1t[i]-prob0[i] for i in range(len(X))])
tpowPP_rs = np.array([prob2t[i]-prob0[i] for i in range(len(X))])

X = np.linspace(0,1,10)
Y = np.linspace(0,1,10)
xgrid, ygrid = np.meshgrid(X, Y)
X = xgrid.flatten()
Y = ygrid.flatten()

I = [i/5-1 for i in range(11)]
ctr_f = griddata((X, Y), tpowPP_ga, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,I,cmap=cm.coolwarm) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('p_0')
plt.ylabel('p_1')
plt.savefig('3c.png')
plt.close()

I = [i/5-1 for i in range(11)]
ctr_f = griddata((X, Y), tpowUX_ga, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,I,cmap=cm.coolwarm) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('p_0')
plt.ylabel('p_1')
plt.savefig('3d.png')
plt.close()

I = [i/250-0.02 for i in range(11)]
ctr_f = griddata((X, Y), tpowPP_rs, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,I,cmap=cm.ocean) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('p_0')
plt.ylabel('p_1')
plt.savefig('3e.png')
plt.close()

I = [i/250-0.02 for i in range(11)]
ctr_f = griddata((X, Y), tpowUX_rs, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,I,cmap=cm.ocean) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('p_0')
plt.ylabel('p_1')
plt.savefig('3f.png')
plt.close()