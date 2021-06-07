#import matplotlib.pyplot as plt
from plt import *


import numpy as np

s=[3.10,2.7,3.2]




f=np.load("output.npz")
x,y=f["x"],f["y"]
x=x[0,:,0]
y=y[0]

x-=np.mean(x)
x/=np.std(x)

plt.plot(x,label="x",alpha=0.5)

for i in range(y.shape[1]):
    arr=np.abs(y[:,i]-1)
    arr[:500]=np.zeros_like(arr)[:500]
    arr/=np.std(arr)
    print(np.argmax(arr))
    topl=[zx if zw>s[i] else 0.0 for zx,zw in zip(x,arr)]
    topl=[j for j,zw in enumerate(arr) if zw>s[i]]
    #plt.plot(topl,alpha=0.2,label=str(i))
    plt.plot(topl,[i for zw in topl],"o",alpha=0.5,label=str(i))

plt.xlabel("timestep")
plt.ylabel("value")

plt.legend()
plt.savefig("output3.png",format="png")
plt.savefig("output3.pdf",format="pdf")
plt.show()
plt.close()

