#import matplotlib.pyplot as plt
from plt import *


import numpy as np

s=[3.10,2.7,3.2]
s=[3.13,2.96,3.4]




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
    topl=[np.argmax(arr)]
    #plt.plot(topl,alpha=0.2,label=str(i))

    plt.plot(topl,[i for zw in topl],"o",markersize=10,alpha=1.0,label=str(i))


plt.xlabel("timestep")
plt.ylabel("value")


plt.legend()
plt.savefig("output4.png",format="png")
plt.savefig("output4.pdf",format="pdf")
plt.show()
plt.close()

