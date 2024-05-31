import pandas as pd
import numpy as np


data = pd.read_csv("data/weather.csv")
print(data)

d = np.array(data)[:,:-1]
print("The attributes are: \n",d)

target = np.array(data)[:,-1]
print("The target is: ",target)
 
def train(c,t):
    specific_hypothesis =[]
    for i, val in enumerate(t):
        if val == "Yes":
            specific_hypothesis = c[i].copy()
            break
    print("after part 1 the specific is ", specific_hypothesis)
    
    for i, val in enumerate(c):
        # print(i,t[i],val)
        if t[i] == "Yes":
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                else:
                    pass
                print(i,t[i],val,x,specific_hypothesis)
                 
                 
    return specific_hypothesis
 
print("The final hypothesis is:",train(d,target))