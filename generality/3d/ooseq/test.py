import numpy as np
import sys

addon=""
if len(sys.argv)>1:
    addon="_"+sys.argv[1]

print(addon)

f=np.load(f"data{addon}.npz")

x,y=f["x"],f["y"]

dex=0

x=x[dex]
y=y[dex]

import matplotlib.pyplot as plt

#plt.plot(x[:,0],color="blue",alpha=0.5,label="x")
for i in range(y.shape[1]):
    plt.plot(y[:,i],label=str(i),alpha=0.5)

plt.legend()

plt.savefig(f"trafo{addon}.png",format="png")


print(np.corrcoef(np.transpose(y[100:])))




