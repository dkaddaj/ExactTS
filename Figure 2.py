# Code to generate Figure 2

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

def rRS(N,p_0,p_1,c):
    out = 0
    for count in range(100000):
        temp = TS_rs(N,p_0,p_1)
        i = temp[0]
        j = temp[1]
        k = temp[2]
        l = temp[3]
        if p_0 < p_1:
            out += (1-F[i][j][k][l]>=c)/100000
        else:
            out += (F[i][j][k][l]>=c)/100000
    return out

P = [i/50 for i in range(51)]
UX = [test_exact(N,p,p,0.99753) for p in P]
gUX = [gest_exact(N,p,p,0.99753) for p in P]
rUX = [rRS(N,p,p,0.99753) for p in P]
PP6 = [test_exact(N,p,p,0.9894462806660532) for p in P]
gPP6 = [gest_exact(N,p,p,0.9894462806660532) for p in P]
rPP6 = [rRS(N,p,p,0.9894462806660532) for p in P]

plt.plot(P,PP6)
plt.plot(P,gPP6)
plt.plot(P,UX)
plt.plot(P,gUX)
plt.legend(['PP(0.6)','PP(0.6)_GA','UX','UX_GA'])
plt.xlabel('p')
plt.ylabel('Type I error')
plt.savefig('2a.png')
plt.close()

plt.plot(P,[PP6[i]-rPP6[i] for i in range(51)])
plt.plot(P,[UX[i]-rUX[i] for i in range(51)])
plt.legend(['PP(0.6)','UX'])
plt.xlabel('p')
plt.ylabel('Difference in type I error')
plt.savefig('2b.png')
plt.close()

uprob0 = [test_exact(200,X[i],Y[i],0.99753) for i in range(len(X))]
uprob1 = [gest_exact(200,X[i],Y[i],0.99753) for i in range(len(X))]
uprob2 = [rRS(200,X[i],Y[i],0.99753) for i in range(len(X))]
prob0 = [test_exact(200,X[i],Y[i],0.9894462806660532) for i in range(len(X))]
prob1 = [gest_exact(200,X[i],Y[i],0.9894462806660532) for i in range(len(X))]
prob2 = [rRS(200,X[i],Y[i],0.9894462806660532) for i in range(len(X))]

powUX_ga = np.array([uprob1[i]-uprob0[i] for i in range(len(X))])
powUX_rs = np.array([uprob2[i]-uprob0[i] for i in range(len(X))])
powPP_ga = np.array([prob1[i]-prob0[i] for i in range(len(X))])
powPP_rs = np.array([prob2[i]-prob0[i] for i in range(len(X))])

X = np.linspace(0,1,10)
Y = np.linspace(0,1,10)
xgrid, ygrid = np.meshgrid(X, Y)
X = xgrid.flatten()
Y = ygrid.flatten()

I = [i/20-0.5 for i in range(13)]
ctr_f = griddata((X, Y), powPP_ga, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,I,cmap=cm.coolwarm) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('p_0')
plt.ylabel('p_1')
plt.savefig('2c.png')
plt.close()

I = [i/1000-0.004 for i in range(9)]
ctr_f = griddata((X, Y), powPP_rs, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,I,cmap=cm.ocean) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('p_0')
plt.ylabel('p_1')
plt.savefig('2d.png')
plt.close()

I = [i/20-0.5 for i in range(13)]
ctr_f = griddata((X, Y), powUX_ga, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,I,cmap=cm.coolwarm) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('p_0')
plt.ylabel('p_1')
plt.savefig('2e.png')
plt.close()

I = [i/1000-0.004 for i in range(9)]
ctr_f = griddata((X, Y), powUX_rs, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,I,cmap=cm.ocean) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('p_0')
plt.ylabel('p_1')
plt.savefig('2f.png')
plt.close()