import matplotlib.pyplot as plt

import numpy as np

f=np.load("output.npz")
x,y=f["x"],f["y"]
x=x[0,:,0]
y=y[0]

x-=np.mean(x)
x/=np.std(x)

plt.plot(x,label="x",alpha=0.5)

for i in range(y.shape[1]):
    arr=np.abs(y[:,i]-1)
    arr[:100]=np.zeros_like(arr)[:100]
    arr/=np.std(arr)
    print(np.argmax(arr))
    plt.plot(arr,label=str(i))

plt.xlabel("timestep")
plt.ylabel("value")

plt.legend()
plt.savefig("output2.png",format="png")
plt.savefig("output2.pdf",format="pdf")
plt.close()

