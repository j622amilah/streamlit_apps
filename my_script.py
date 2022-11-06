# Created by Jamilah Foucher, le 18 avril 2022

# une fonction qui trie un vecteur
import numpy as np

def sortarr(arr):
    sarr = []
    temp = arr
    while(len(temp) > 1):
        mv = min(temp)
        cnt = [1 for r in temp if r == mv]
        vals = [mv for r in range(sum(cnt))]
        sarr.append(vals)
        temp = [q for q in temp if q != mv]
    if any(temp) == True:
        sarr.append(temp)
        
    # une liste avec les valeurs unique
    unq = [sarr[i][0] for i in range(len(sarr))]
    
    # une liste avec les valeurs qui repeter
    nonunq = []
    ind = []
    for i in unq:
        for indd, j in enumerate(arr):
            if i == j:
                nonunq.append(j)
                ind.append(indd)
    
    return ind, unq, nonunq


A = [2, 4, 6, 4, 3]
A = np.array(A)
print('A : ', A)

out = sortarr(A)
print('sorted A:', out)
