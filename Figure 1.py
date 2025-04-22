# Code to generate Figure 1

import scipy.special as sc
from scipy.stats import norm
from scipy.integrate import dblquad
import time
import numpy as np
import matplotlib.pylab as plt
import math
import pickle
from scipy.stats import binom
from matplotlib import cm
from scipy.interpolate import griddata

def integer_exact(i,j,k,l):
    return sum([sc.beta(i+m+1,j+l+2)/((l+m+1)*sc.beta(m+1,l+1)*sc.beta(i+1,j+1)) for m in range(k+1)])

def gauss_approx(i,j,k,l):
    mu_x = (i+1)/(i+j+2)
    mu_y = (k+1)/(k+l+2)
    sig_x = (i+1)*(j+1)/(((i+j+2)**2)*(i+j+3))
    sig_y = (k+1)*(l+1)/(((k+l+2)**2)*(k+l+3))
    return norm.cdf((mu_y-mu_x)/((sig_x+sig_y)**0.5))

def sample_approx(iter,i,j,k,l):
    out = 0
    for count in range(iter):
        x = np.random.beta(i+1,j+1)
        y = np.random.beta(k+1,l+1)
        out += (y>x)/iter
    return out

def integral_approx(i,j,k,l):
    return 1-dblquad(lambda x, y: ((x**i)*((1-x)**j)*(y**k)*((1-y)**l))/(sc.beta(i+1,j+1)*sc.beta(k+1,l+1)), 0, 1, lambda x: x, lambda x: 1)[0]

Exact = []
GA = []
RS = []
NI = []
for n in range(1,251):
    start = time.time()
    for i in range(100):
        temp = integer_exact(n,n,n,n)
    end = time.time()
    Exact.append((end-start)/100)
    start = time.time()
    for i in range(100):
        temp = gauss_approx(n,n,n,n)
    end = time.time()
    GA.append((end-start)/100)
    start = time.time()
    for i in range(100):
        temp = sample_approx(10000,n,n,n,n)
    end = time.time()
    RS.append((end-start)/100)
    start = time.time()
    for i in range(100):
        temp = integral_approx(n,n,n,n)
    end = time.time()
    NI.append((end-start)/100)
N = [4*n for n in range(1,251)]
plt.plot(N, Exact)
plt.plot(N, GA)
plt.plot(N, RS)
plt.plot(N, NI)
plt.yscale('log')
plt.legend(['Exact','GA','RS','NI'])
plt.xlabel('N')
plt.ylabel('computation time')
plt.savefig('1a.png')
plt.close()

with open ('F_200.txt', 'rb') as file:
    F = pickle.load(file)
with open ('G_200.txt', 'rb') as file:
    G = pickle.load(file)

def rs_err(a,b,c,d):
    p = min(1,max(0,1-F[a][b][c][d]))
    l = int(10000*p)+1
    return binom.pmf(l-1,9999,p)*2*p*(1-p)

def paired_rs(a,b):
    avg = 0
    count = 0
    for i in range(a+1):
        for j in range(b+1):
            count += 1
            avg += rs_err(i,a-i,j,b-j)
    return avg/count

def paired_max(a,b):
    biggest = 0
    for i in range(a+1):
        for j in range(b+1):
            comp = abs(F[i][a-i][j][b-j]-G[i][a-i][j][b-j])
            if comp > biggest:
                print(i,j)
                biggest = comp
    return biggest

def paired_avg(a,b):
    avg = 0
    count = 0
    for i in range(a+1):
        for j in range(b+1):
            count += 1
            avg += abs(F[i][a-i][j][b-j]-G[i][a-i][j][b-j])
    return avg/count

X = [i for i in range(0,101,2)]
Y = [i for i in range(0,101,2)]
xgrid, ygrid = np.meshgrid(X, Y)
X = xgrid.flatten()
Y = ygrid.flatten()
Z_max = [paired_rs(X[i],Y[i]) for i in range(len(X))]
ctr_f = griddata((X, Y), Z_max, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,cmap=cm.coolwarm) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('a+b')
plt.ylabel('c+d')
plt.savefig('1b.png')
plt.close()

X = [i for i in range(0,101)]
Y = [i for i in range(0,101)]
xgrid, ygrid = np.meshgrid(X, Y)
X = xgrid.flatten()
Y = ygrid.flatten()
Z_max = [paired_max(X[i],Y[i]) for i in range(len(X))]
ctr_f = griddata((X, Y), Z_max, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,cmap=cm.ocean) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('a+b')
plt.ylabel('c+d')
plt.savefig('1c.png')
plt.close()

X = [i for i in range(0,101)]
Y = [i for i in range(0,101)]
xgrid, ygrid = np.meshgrid(X, Y)
X = xgrid.flatten()
Y = ygrid.flatten()
Z_avg = [paired_avg(X[i],Y[i]) for i in range(len(X))]
ctr_f = griddata((X, Y), Z_avg, (xgrid, ygrid), method='linear')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
proj = ax.contourf(xgrid, ygrid, ctr_f,cmap=cm.PuOr) 
fig.colorbar(proj, shrink=0.5, aspect=5)
plt.xlabel('a+b')
plt.ylabel('c+d')
plt.savefig('1d.png')
plt.close()