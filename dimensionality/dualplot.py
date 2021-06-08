import numpy as np
import os
from simplestat import statinf

from plt import *

f,fa=np.load("fullrun/result.npz"),np.load("weakrun/result.npz")
x,y,s=f["x"],f["y"],f["s"]
xa,ya,sa=fa["x"],fa["y"],fa["s"]

plt.errorbar(x,y,fmt="o",yerr=s,label="6300 Samples")
plt.errorbar(xa,ya,fmt="o",yerr=sa,alpha=0.8,label="500 Samples")

plt.legend()

plt.xlabel("dimension")
plt.ylabel("AUC")

plt.savefig("cdimensional.png",format="png")
plt.savefig("cdimensional.pdf",format="pdf")


plt.show()

for xx,yy,ss in zip(x,y,s):
    print(f"{xx}:{yy}+-{ss}")







