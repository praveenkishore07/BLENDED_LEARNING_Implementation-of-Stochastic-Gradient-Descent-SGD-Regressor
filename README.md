# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm-

Import necessary libraries (pandas, numpy, sklearn, matplotlib).

Load the dataset using pandas.

Preprocess the data:

Drop unnecessary columns (e.g., CarName, car_ID).
Convert categorical variables using one-hot encoding.
Split the dataset into features (X) and target (Y), then into training and testing sets.

Standardize the features and target using StandardScaler.

Initialize the SGDRegressor model with appropriate parameters.

Train the model on the training data.

Predict the target values for the test data.

Evaluate the model using Mean Squared Error and R² score.

Display model coefficients and intercept.

Visualize actual vs predicted values with a scatter plot.

End of workflow.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: A PRAVEEN KISHORE
RegisterNumber:  212225220074


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error , r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)

X=data.drop('price',axis=1)
y=data['price']

scaler = StandardScaler()
X= scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

sgd_model=SGDRegressor(max_iter=1000 ,tol=1e-3)

sgd_model.fit(X_train,y_train)

y_pred=sgd_model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("Name: A PRAVEEN KISHORE")
print("Reg.no : 212225220074")
print()
print(f"MSE : {mse}")
print(f"R2 : {r2}")
print(f"MAE : {mae}")

print("\nModel Cofficients: ")
print("Cofficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()
*/
```

## Output:
## LOAD THE DATASET--
<img width="997" height="600" alt="1" src="https://github.com/user-attachments/assets/56fcf495-835b-40c7-b401-d4eb9f17dbd8" />
<img width="597" height="757" alt="2" src="https://github.com/user-attachments/assets/013b3cee-b72e-44ca-843d-177f493247c7" />

## Residuals -
<img width="317" height="131" alt="3" src="https://github.com/user-attachments/assets/31b838c7-9a68-4e4e-8756-dbdef56c30af" />

## Model cofficients -
<img width="865" height="237" alt="4" src="https://github.com/user-attachments/assets/bfb1dabf-1319-4662-a920-7716ebf550b4" />
## Visualization Graph -
<img width="865" height="577" alt="5" src="https://github.com/user-attachments/assets/cb5cb8c1-a9d7-4bcc-8d6d-4c1ba1e88451" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
