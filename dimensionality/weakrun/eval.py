import numpy as np
import os
from simplestat import statinf

from plt import *


fns=["results/"+zw for zw in os.listdir("results")]

def tri(q):
    try:
        print("tryin")
        ret= q()
        print("tryout")
        return ret
    except:
        return None


fs=[np.load(fn) for fn in fns if np.random.random()<0.95]
fs=[zw for zw in fs if not zw is None]

print("loading",len(fs),"files")


aucs=[float(f["auc"]) for f in fs]
outdims=[int(f["outdim"]) for f in fs]
dms=[int(f["dm"]) for f in fs]

q={}

for auc,outdim,dm in zip(aucs,outdims,dms):
    key="_d" if dm else ""
    key=str(outdim)+key
    if not key in q.keys():q[key]=[]
    q[key].append(auc)




x=list(q.keys())
x=[zw.replace("_d","") for zw in x if not "d" in zw]
y=[np.mean(q[xx]) for xx in x]
s=[np.std(q[xx])/np.sqrt(len(q[xx])) for xx in x]


x=[int(zw) for zw in x]

plt.errorbar(x,y,fmt="o",yerr=s)
np.savez_compressed("result",x=x,y=y,s=s)



plt.xlabel("dimension")
plt.ylabel("AUC")

plt.savefig("dimensional.png",format="png")
plt.savefig("dimensional.pdf",format="pdf")


plt.show()

for xx,yy,ss in zip(x,y,s):
    print(f"{xx}:{yy}+-{ss}")







