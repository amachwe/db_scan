import numpy as np
import pandas as pd

def nbrs(a,ds,eps):
    __nbrs = []
    dist = []
    
    for idx,b in enumerate(ds):
        d = np.linalg.norm(b-a)
        
        if d <= eps:
            dist.append(d)
            __nbrs.append(idx)
        
    return __nbrs, dist

def dbscan(df, eps=3, numpts = 10, axis=0):
    
    ds = df.to_numpy()
    
    c = 0
    ia = 0
    len_ds = len(ds)
    label = np.full(len_ds,-2)
    
    
    for idx, ia in enumerate(ds):
        
        if label[idx]>-2:
            continue
        
        _nbrs,d = nbrs(ia,ds,eps)
            
        if len(_nbrs)+1 < numpts:
            label[idx] = -1
            continue
        
        label[idx] = c
        seedset = _nbrs
        
        for s in seedset:
            
            if label[s] == -1:
                label[s]=c
            if label[s] > -2:
                continue
            
            label[s] = c
            _nbrs, _ = nbrs(ds[s],ds,eps)
            
            if len(_nbrs)+1 >= numpts:
                
                ss = set(seedset)
                for i in _nbrs:
                    if i not in ss:
                        seedset.append(i)

        c = c+1
    
    return ds,label


    