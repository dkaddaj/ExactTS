import pickle
import math
from scipy.stats import norm

with open ('betaln_vals_1000.txt', 'rb') as file:
    betaln = pickle.load(file)

def gauss_approx(i,j,k,l):
    mu_x = (i+1)/(i+j+2)
    mu_y = (k+1)/(k+l+2)
    sig_x = (i+1)*(j+1)/(((i+j+2)**2)*(i+j+3))
    sig_y = (k+1)*(l+1)/(((k+l+2)**2)*(k+l+3))
    return norm.cdf((mu_y-mu_x)/((sig_x+sig_y)**0.5))

N = 200
F = [[[[sum([math.exp(betaln[i+m][j+l+1]-betaln[m][l]-betaln[i][j]-math.log((l+m+1))) for m in range(k+1)]) for l in range(N+1-i-j-k)] for k in range(N+1-i-j)] for j in range(N+1-i)] for i in range(N+1)]
G = [[[[gauss_approx(i,j,k,l) for l in range(N+1-i-j-k)] for k in range(N+1-i-j)] for j in range(N+1-i)] for i in range(N+1)]

w = [[[[0 for l in range(N+1-i-j-k)] for k in range(N+1-i-j)] for j in range(N+1-i)] for i in range(N+1)]
w[0][0][0][0] = 1
for n in range(1,N+1):
    for i in range(n+1):
        for j in range(n+1-i):
            for k in range(n+1-i-j):
                l = n-i-j-k
                if i>0:
                    w[i][j][k][l] += w[i-1][j][k][l]*F[i-1][j][k][l]
                if j>0:
                    w[i][j][k][l] += w[i][j-1][k][l]*F[i][j-1][k][l]
                if k>0:
                    w[i][j][k][l] += w[i][j][k-1][l]*(1-F[i][j][k-1][l])
                if l>0:
                    w[i][j][k][l] += w[i][j][k][l-1]*(1-F[i][j][k][l-1])
w_ga = [[[[0 for l in range(N+1-i-j-k)] for k in range(N+1-i-j)] for j in range(N+1-i)] for i in range(N+1)]
w_ga[0][0][0][0] = 1
for n in range(1,N+1):
    for i in range(n+1):
        for j in range(n+1-i):
            for k in range(n+1-i-j):
                l = n-i-j-k
                if i>0:
                    w_ga[i][j][k][l] += w_ga[i-1][j][k][l]*G[i-1][j][k][l]
                if j>0:
                    w_ga[i][j][k][l] += w_ga[i][j-1][k][l]*G[i][j-1][k][l]
                if k>0:
                    w_ga[i][j][k][l] += w_ga[i][j][k-1][l]*(1-G[i][j][k-1][l])
                if l>0:
                    w_ga[i][j][k][l] += w_ga[i][j][k][l-1]*(1-G[i][j][k][l-1])


with open('F_200.txt', 'wb') as file:
    pickle.dump(F, file)                    
with open('w_200.txt', 'wb') as file:
    pickle.dump(w, file)
with open('w_ga_200.txt', 'wb') as file:
    pickle.dump(w_ga, file)