import os

outdims=[1,2,3,4,5,6,7,8,9,10,12,15,20,25,50]
dms=[*[1 for i in outdims],*[0 for i in outdims]]
dms=[0 for i in outdims]

for i in range(0,100000):
    outdim=outdims[i%len(outdims)]
    dm=dms[i%len(dms)]

    os.system(f"python3 main.py {outdim} {dm} {i}")










