# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 08:35:11 2019

@author: bhavy
"""
import pandas as pd 
import datetime
import geopy
import numpy as np
check=pd.read_csv("df142.csv")

def clean_check(test):
    test["date"]=datetime.datetime.strptime(test["date"],"%Y-%m-%d %H:%M:%S.%f")
#    test['time_month'] = test['date'].month
    test['time_hour'] = test['date'].hour
    test['time_mintues'] = test['date'].minute
    test.drop(test.index[[0,1,2,3,4,5,7,9,10,11,12,13,14]],inplace=True)

    route=pd.read_csv("df_test_clean.csv")
    cor1=(test["latitude"],test["longitude"])
    min_d=[(99999,0)]
    for j in range(len(route)):
        dis=geopy.distance.geodesic(cor1,(route.loc[j]["latitude"],route.loc[j]["longitude"])).km
        if dis<0.5:
            min_d=[(dis,j)]
            break
        else:
            if min_d[0][0]>dis:
                min_d=[(dis,j)]
    k=min_d[0][1]
    route.drop(labels="location",inplace=True,axis=1)
    li=[]

    for abc in route.loc[k]:
        li.append(abc)       
    for abc in test:
        li.append(abc)
    myArray=np.array(li)
    return myArray.reshape(1,9)

import pickle 
pickle_in=open("regressor_month.pickle","rb")
reg1=pickle.load(pickle_in)


test=check.loc[5]
test1=check.loc[50]
test2=check.loc[500]
test3=check.loc[1100]
test4=check.loc[1600]
test5=check.loc[25]
check["date"]=pd.to_datetime(check["date"])

output=check.loc[len(check)-1]["date"]-check.loc[5]["date"]
output1=check.loc[len(check)-1]["date"]-check.loc[50]["date"]
output2=check.loc[len(check)-1]["date"]-check.loc[500]["date"]
output3=check.loc[len(check)-1]["date"]-check.loc[1100]["date"]
output4=check.loc[len(check)-1]["date"]-check.loc[1600]["date"]
output5=check.loc[len(check)-1]["date"]-check.loc[25]["date"]
output=output.days*1440+(output.seconds)/60.0
output1=output1.days*1440+(output1.seconds)/60.0
output2=output2.days*1440+(output2.seconds)/60.0
output3=output3.days*1440+(output3.seconds)/60.0
output4=output4.days*1440+(output4.seconds)/60.0
output5=output5.days*1440+(output5.seconds)/60.0


ar=clean_check(test)
out=reg1.predict(ar)
ar=clean_check(test1)
out1=reg1.predict(ar)
ar=clean_check(test2)
out2=reg1.predict(ar)
ar=clean_check(test3)
out3=reg1.predict(ar)
ar=clean_check(test4)
out4=reg1.predict(ar)
ar=clean_check(test5)
out5=reg1.predict(ar)


    
    
    
    
        
        