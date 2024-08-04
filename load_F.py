import pickle
import math
with open ('betaln_vals_1000.txt', 'rb') as file:
    betaln = pickle.load(file)
N = 200
F = [[[[sum([math.exp(betaln[i+m][j+l+1]-betaln[m][l]-betaln[i][j]-math.log((l+m+1))) for m in range(k+1)]) for l in range(N+1-i-j-k)] for k in range(N+1-i-j)] for j in range(N+1-i)] for i in range(N+1)]
with open('F_200.txt', 'wb') as file:
    pickle.dump(F, file)