from tensorflow.keras import backend as K

def loss(outdim,self=False,K=K):
    """orthogonal oneoff loss
are sadly not trivial to explain. I accidentally created it in one of my autoencoders (see my masters thesis), and it took me another O(100) networks to notice that this could be useful, and to this day another O(1000) to understand them. And this is the version of increased complexity that also allows you to train orthogonal oneoffs.

First the parameters: outdim is the number of orthogonal predictions. Complexity grows ~outdim**2, so dont overdo it (100 is usually already to much, especcially since there is no guarantuee for orthogonality and it still depends heavily on what your network can do/how good your training is, 3 seems to usually be a good first choice)
self is a parameter that increases complexity (~outdim) but can make the network focus more on the anomaly prediction and less on the orthogonality (oo instead of ooo)
K: allows you to switch the backend in a crude way. Not adding anthing just uses keras


How to use this?
model.compile(...,loss=loss(outdim),...)
and then model.fit(input_data,input_data,....) (the second data just does not get used)
or model.fit(input_data,.....)
and add use_bias=False to each layer
arbitrary dimensional input, but the last dimension should be [?,...,outdim]



Now to quickly explain oneoffs, lets look first at the diagonal parts of the loss

loss=(x-1)**2

As you migth notice, this tries to set x=1, why is that anomalous?
Lets look at two inputs: 
    if the input is constant, the network will divide by this constant and the outout will be one. Anything nonconstant is anomalous and wont be mapped off one (thats why oneoffs btw)
    now for a more complicated function. Lets think there are two inputs, which are always 1 apart from each other. The network will just subtract them, an everything different apart will be mapped off one, and again the anomaly detection works.
These example are the trivial cases, but you can show, that for arbitrary functions (without a constant taylor member close to 0) you can define them through a oneoff network, and thus you can find outliers to this function using oo's. Also you can show, that oneoffs provide a good estimate of how to combine two features, which makes them also viable on complicated data.
There are a lot more details about oo's that I ignore here, but you can look up in my masters thesis, which you best find through my CV (http://www.psorus.de/s/cv.html).


Now for orthogonal oo's.
The loss becomes something like
sum_i(sum_j[j>i](abs(mean((x_i-1)*(x_j-1)))))

You still see the same force pushing x_i->1, but there is a secondary effect: if you define y_i=x_i-1 this becomes similar to a covariance coefficiant, which is minimal at uncorrelated events. This should make clear why this creates orthogonal predictions, but I want to quickly highlight how fragil this formula is in machine learning. For example correlations will have divergent gradients, switching abs and mean make the orthogonality disappear and replacing the > with a >= (like self=True) does, can kill the orthogonality depending on the data used.





    """
    def lss(a,b):
        q=b
        pd=[i for i in range(len(q.shape))]
        pd.remove(pd[-1])
        pd.insert(0,len(pd))
        #print(pd)
        q=K.permute_dimensions(q,tuple(pd))
        #exit()

        #print(q.shape)

        adl=None

        for i in range(outdim):
            for j in range(i+(0 if self else 1),outdim):
              ac=K.abs(K.mean(((q[i]-1)*(q[j]-1))))
              if adl is None:
                  adl=ac
              else:
                  adl+=ac

        return adl
    return lss
