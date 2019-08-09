# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:58:24 2019

@author: bhavy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:58:28 2019

@author: bhavy
"""
from sklearn.metrics import mean_squared_error
import statistics
from math import sqrt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
import test_final_data
sns.set(style="white")
data=pd.read_csv("250km_route.csv")

data=test_final_data.df_clean(data)

#data=test_final_data.df_super_clean(data)
#    data.drop(data.columns[[13,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]],axis=1,inplace=True)
#corr = data.iloc[:,[0,1,2,3,4,5,6,10,14,16,17,18,19,20,21,22,23,25,25,26,27,28,29,30,31,32,33,34,35]].corr()
corr=data.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})



#X=data.drop(labels=["time_to_complete","end_time"],axis=1).values

# For no change in dataset 
X=data.iloc[:,[0,1,2,3,4,5,6,17,18]].values

y=data["time_to_complete"].values

reg_ols=sm.OLS(endog=y,exog=X).fit()
reg_ols.summary()



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size = 0.7, random_state = 4)



#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
##sy = StandardScaler()
##y_train = sy.fit_transform(y_train.reshape(-1,1))
##y_test = sy.transform(y_test.reshape(-1,1))


##### Testing 28k from different month 
#test=pd.read_csv("test_df_28k.csv")
#test=test_final_data.df_clean(test)
#X=test.drop(labels=["time_to_complete","end_time"],axis=1).values
#y=test["time_to_complete"].values


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 30, random_state = 0)
regressor.fit(X_train,y_train)
accuracy = regressor.score(X_dev,y_dev)
y_test2=regressor.predict(X_dev)

y_prec=(y_test2-y_dev)/y_dev

y_prec1=[]
for abc in y_prec:
    if abc>=0:
        y_prec1.append(abc)
    else:
        y_prec1.append(-abc)
        
count=0
for abc in y_prec1:
    if abc>0.5:
        count+=1        
print(count)
print(count*100/len(y_dev))
        
sns.scatterplot(y_dev,y_test2)


# Plot Trees count 
#a = np.array([[10, 247]])
#for i in range(20, 60, 10):
#    clf = RandomForestRegressor(n_estimators = i)
#    clf.fit(X_train, y_train)
#    y_actual = y_test
#    y_pred = clf.predict(X_test)
#    rms = sqrt(mean_squared_error(y_actual, y_pred))
#    a = np.append(a, [[i, rms]], axis = 0)
#    
#plt.plot(a[:, 0], a[:, 1], linewidth = 2.0)
#plt.title('RMSE VS No. of Trees')
#plt.xlabel('No. of Trees')
#plt.ylabel('RMSE')
#plt.show()
import pickle
pickle_out=open("regressor_month_250k.pickle","wb")
pickle.dump(regressor,pickle_out)



