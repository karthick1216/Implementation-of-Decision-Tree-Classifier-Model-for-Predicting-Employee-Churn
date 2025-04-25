# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

## step 1: Import Libraries and Load Dataset.
## step 2: Preprocess the Data.
## step 3: Split the Dataset.
## step 4: Train the Decision Tree Classifier.
## step 5: Make Predictions and Evaluate the Model

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: KARTHICK S

RegisterNumber: 212224230114
*/

```
import pandas as pd
data=pd.read_csv("/content/Employee (1).csv")
data.head()
```
## output:
![ex 8 s1](https://github.com/user-attachments/assets/fdd15a48-7ba7-4034-9119-44f5a3871f1d)

```
data.info()
data.isnull().sum()
data['left'].value_counts()
```
## output:
![ex 8 s2](https://github.com/user-attachments/assets/96195601-6a32-4a12-b2af-482c03c40ef5)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()
```
## output:
![ex 8 s3](https://github.com/user-attachments/assets/ce1f81a5-f89c-4d83-b8de-33bc219b7995)

```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```
## output:
![ex 8 s4](https://github.com/user-attachments/assets/0d0e5669-7119-4ca0-b0f2-db5d0cd0b9b9)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
