# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:43:34 2023

@author: lenovo
"""


1]PROBLEM

BUSINESS OBJECTIVE==We are going to predict PROFIT using different attributes..



#Importing the Necessary Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt
import scipy
from scipy import stats
import pylab

#Loading the Dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Multiple Linear Regression/50_Startups.csv')

#EDA
df.info()
df.describe()
df.drop(['State'],axis=1,inplace=True)
df=df.rename(columns={'R&D Spend':'rdspend','Administration':'admin','Marketing Spend':'marketspend','Profit':'profit'})

#Graphical Visulization(Univariate)

#rdspend
plt.hist(df.rdspend)
plt.boxplot(df.rdspend)
plt.bar(height=df.rdspend,x=np.arange(1,51,1))

#admin
plt.hist(df.admin)
plt.boxplot(df.admin)
plt.bar(height=df.admin,x=np.arange(1,51,1))

#marketspend
plt.hist(df.marketspend)
plt.boxplot(df.marketspend)
plt.bar(height=df.marketspend,x=np.arange(1,51,1))

#profit
plt.hist(df.profit)
plt.boxplot(df.profit)
plt.bar(height=df.profit,x=np.arange(1,51,1))

#There are Outliers in profit.so we replace it by logical value using Winsorizer function 
from feature_engine.outliers import Winsorizer
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['profit'])
df['profit']=w.fit_transform(df[['profit']])

#Scatter plot along with Histogram
sns.pairplot(df)

#Correlation Matrix
df.corr()
#After Analyzing,there is colineratiy problem exists between input variables such as (marketspend and rdspend)

#Preparing the model
import statsmodels.formula.api as smf
model=smf.ols('profit ~ rdspend+admin+marketspend',data=df).fit()
model.summary()
#P values of admin,marketspend are more than 0.05

#Checking whether any influential values
#Influential Index plot
import statsmodels.api as sm
sm.graphics.influence_plot(model)

#Index 49,48,46,45 have high influence value,so we are going to exclude entire row.
df1=df.drop(df.index[[49,48,46,45]])

#again prepare the model
model=smf.ols('profit ~ rdspend+admin+marketspend',data=df1).fit()
model.summary()

#Checking Colinearity to decide whether which variables we are going yo remove using VIF.
#Assumption::- VIF>10=Colinearity
#Checking Colinearity for individual variables

rdspend_c=smf.ols('rdspend ~ admin+marketspend+profit',data=df1).fit().rsquared
rdsepnd_value=1/(1-rdspend_c)

admin_c=smf.ols('admin ~ rdspend+marketspend+profit',data=df1).fit().rsquared
admin_value=1/(1-admin_c)

marketspend_c=smf.ols('marketspend ~ admin+rdspend+profit',data=df1).fit().rsquared
marketsepnd_value=1/(1-marketspend_c)

#using admin.. R*2 value get reduced,so we exclude it..

fianl_model=smf.ols('(profit) ~ (marketspend+rdspend)',data=df1).fit()
fianl_model.summary()

pred=fianl_model.predict(df1)

#Probplot for normality
stats.probplot(pred,plot=pylab)

#Q-Qplot for normality
res=fianl_model.resid
sm.qqplot(res)

#Fitted vs Residual Plot
sns.residplot(x=pred,y=df1.profit,lowess=True)
plt.xlabel('fitted')
plt.ylabel('residual')
plt.title('fitted vs residual')
plt.show()

#RMSE
error=df1.profit-pred
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#LOG Transformations
fianl_model=smf.ols('(profit) ~ np.log(marketspend+rdspend)',data=df1).fit()
fianl_model.summary()

#expontial Transformations
fianl_model=smf.ols('np.log(profit) ~ (marketspend+rdspend)',data=df1).fit()
fianl_model.summary()

#Square Transformations
fianl_model=smf.ols('(profit) ~ (marketspend+rdspend)*(marketspend+rdspend)',data=df1).fit()
fianl_model.summary()

#We tune the model using different transformations..but no one gives best result..so we use '(profit) ~ (marketspend+rdspend)' .these transformastion as Final...



from sklearn.model_selection import train_test_split
train,test=train_test_split(df1,test_size=0.2)

model_train=smf.ols('profit ~ marketspend+rdspend',data=train).fit()
model_train.summary()
#FOR TRAIN DATA
train_pred=model_train.predict(train)

# train residual values 
train_resid  = train_pred - train.profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

#FOR TEST DATA
test_pred=model_train.predict(test)
#test residual values
test_resid = test_pred - test.profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse



2]PROBLEM

BUSINESS OBJECTIVE:-Predicting the Price of model using different attributes.



#Loading the dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/360DIGITMG ASSIGNMENT/multiple linear/dataset/ToyotaCorolla.csv',encoding=('ISO-8859-1'))

#As per given in the problem,we are excluding some columns which are not relevant.
df=df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

#EDA
df.info()
df.describe()
df=df.rename(columns={'Price':'price','Age_08_04':'age','Quarterly_Tax':'tax','Weight':'weight','Doors':'doors','Gears':'gear'})

#Graphical Representation(Univarate Data)
plt.boxplot(df.price)
plt.boxplot(df.age)
plt.boxplot(df.KM)
plt.boxplot(df.HP)
plt.boxplot(df.cc)
plt.boxplot(df.doors)
plt.boxplot(df.gear)
plt.boxplot(df.tax)
plt.boxplot(df.weight)
#all columns have outliers,so we remove it by using Winsorizer
from feature_engine.outliers import Winsorizer
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['price'])
df['price']=w.fit_transform(df[['price']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['age'])
df['age']=w.fit_transform(df[['age']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['KM'])
df['KM']=w.fit_transform(df[['KM']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['HP'])
df['HP']=w.fit_transform(df[['HP']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['cc'])
df['cc']=w.fit_transform(df[['cc']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['doors'])
df['doors']=w.fit_transform(df[['doors']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['gear'])
df['gear']=w.fit_transform(df[['gear']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tax'])
df['tax']=w.fit_transform(df[['tax']])
w=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['weight'])
df['weight']=w.fit_transform(df[['weight']])

#Histogram
plt.hist(df.price)
plt.hist(df.age)
plt.hist(df.KM)
plt.hist(df.HP)
plt.hist(df.cc)
plt.hist(df.doors)
plt.hist(df.gear)
plt.hist(df.tax)
plt.hist(df.weight)

#Scatter Diagram(Bivariate Data)
sns.pairplot(df)
#Corelation Matrix
corr=df.corr()

#Regression Model
import statsmodels.formula.api as smf
model=smf.ols('price ~ age+KM+HP+cc+doors+gear+tax+weight',data=df).fit()
model.summary()

pred=model.predict(df)

#LOG Transformations
model=smf.ols('price ~ np.log(age+KM+HP+cc+doors+gear+tax+weight)',data=df).fit()
model.summary()
#EXPONTIAL Transformations
model=smf.ols('np.log(price) ~ age+KM+HP+cc+doors+gear+tax+weight',data=df).fit()
model.summary()
#SQUARE Transformations
model=smf.ols('price ~ age+KM+HP+cc+doors+gear+tax+weight * age+KM+HP+cc+doors+gear+tax+weight',data=df).fit()
model.summary()
#SQUARE ROOT Transformations
model=smf.ols('(price) ~ np.sqrt(age+KM+HP+cc+doors+gear+tax+weight)',data=df).fit()
model.summary()

#Finally we choose SQUARE Transformations

f_model=smf.ols('price ~ age+KM+HP+cc+doors+gear+tax+weight * age+KM+HP+cc+doors+gear+tax+weight',data=df).fit()
f_model.summary()

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm
sm.graphics.influence_plot(f_model)

#some Influential values(rows) we are going to drop

df1=df.drop(df.index[[960,523,1109,1073,696]])

#Develop Final model

f_model=smf.ols('price ~ age+KM+HP+cc+doors+gear+tax+weight * age+KM+HP+cc+doors+gear+tax+weight',data=df1).fit()
f_model.summary()

predictt=f_model.predict(df1)

#Q-Q Plot
res=f_model.resid
stats.probplot(res,plot=pylab)

# Residuals vs Fitted plot
sns.residplot(x = predictt, y = df1.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train,test = train_test_split(df1, test_size = 0.2) # 20% test data

trainmodel=f_model=smf.ols('price ~ age+KM+HP+cc+doors+gear+tax+weight * age+KM+HP+cc+doors+gear+tax+weight',data=df1).fit()
trainmodel.summary()

test_pred = trainmodel.predict(test)

# test residual values 
test_resid = test_pred - test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = trainmodel.predict(train)

# train residual values 
train_resid  = train_pred - train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
