import numpy as np


f=np.load("roc.npz")
fpr,tpr=f["fpr"],f["tpr"]


def iterdual(q):
    las=None
    for zw in q:
        if not las is None:
            yield las,zw
        las=zw
def iterdelta(q):
    for a,b in iterdual(q):
        yield a-b
def itermean(q):
    for a,b in iterdual(q):
        yield (a+b)/2

def integrate(x,y):
    ret=0.0
    for xx,yy in zip(iterdelta(x),itermean(y)):
        ret+=xx*yy
    return ret
    


#print(fpr.shape,tpr.shape)


auc=integrate(fpr,tpr)

print("AUC:      ",auc)

print("1/(1-AUC):",1/(1-auc))







