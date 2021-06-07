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
    plt.plot(y[:,i],label=str(i))

plt.xlabel("timestep")
plt.ylabel("value")

plt.legend()
plt.savefig("output.png",format="png")
plt.savefig("output.pdf",format="pdf")
plt.close()

