import numpy as np
import os
from simplestat import statinf

fns=["results/"+zw for zw in os.listdir("results")]


fs=[np.load(fn) for fn in fns]

aucs=[float(f["auc"]) for f in fs]
outdims=[int(f["outdim"]) for f in fs]

q={}

for auc,outdim in zip(aucs,outdims):
    if not outdim in q.keys():q[outdim]=[]
    q[outdim].append(auc)



from plt import *

x=list(q.keys())
y=[np.mean(q[xx]) for xx in x]
s=[np.std(q[xx])/np.sqrt(len(q[xx])) for xx in x]


plt.xlabel("dimension")
plt.ylabel("auc")

plt.errorbar(x,y,fmt="o",yerr=s)
plt.show()








