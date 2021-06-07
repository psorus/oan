import numpy as np



d2=3744
def load_train():
    x=np.load("keras-example/train.npz")["x"]
    #return x.reshape((x.shape[0]*x.shape[1],x.shape[2]))

    #more than one peak per run of the recurrent network. Else kinda useless
    d1=(x.shape[0]*x.shape[1])
    assert not d1%d2,(d1,d1/d2)
    d1=d1//d2
    
    return x.reshape((d1,d2,x.shape[2]))

def load_anomaly():
    x=np.load("keras-example/anomaly.npz")["x"][:d2]
    addon=np.zeros_like(x)
    a1,a2=1100,1250
    addon[a1:a2]=np.abs(np.sin(np.expand_dims(np.arange(a1,a2,1,dtype="float"),1)*0.1)*25)
    return (x+addon).reshape(1,d2,1)



if __name__=="__main__":
    train=load_train()
    anomaly=load_anomaly()

    from plt import *

    plt.plot(anomaly[0,:,0])
    plt.show()


