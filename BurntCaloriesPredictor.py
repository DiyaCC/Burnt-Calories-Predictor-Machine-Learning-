import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# Inspiration from https://www.geeksforgeeks.org/calories-burnt-prediction-using-machine-learning/

######################################################################

df1 = pd.read_csv('calories.csv')
df2 = pd.read_csv('exercise.csv')

# Combining both datasets

df = pd.read_csv('exercise.csv')
df['Calories'] = df1['Calories']


# Exploratory data analysis (using visuals to understand data)

sb.scatterplot(x=df['Height'], y=df['Weight'])

features = ['Age', 'Height', 'Weight', 'Duration']

plt.subplots(figsize = (15,10) )

for i, col in enumerate(features):
    plt.subplot(2, 2, i+1)
    x = df.sample(1000)
    sb.scatterplot(x = x[col], y = x['Calories'])

plt.tight_layout()

features = df.select_dtypes(include = 'float').columns

plt.subplots(figsize = (15, 10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.distplot(x = df[col], axlabel= col )

plt.tight_layout()
plt.show()

# making male and female into 0 and 1, respectively

df.replace({'male':0, 'female': 1}, inplace = True)

# removing columns weight and duration

remove = ['Weight', 'Duration']

df.drop(remove, axis = 1, inplace = True)

# Model training

features = df.drop(['User_ID', 'Calories'], axis = 1) # what is left are all x values. UserID was unneeded so removed

target = df['Calories'].values # target is the y-value that we want to predict

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.1, random_state = 22)


# normalizing the data

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# predicting

from sklearn.metrics import mean_absolute_error as mae

models = [LinearRegression(), XGBRegressor(), Lasso(), RandomForestRegressor(), Ridge()]

i = 0

while i < len(models):
    models[i].fit(x_train, y_train) # trains the model

    predicted_test = models[i].predict(x_test)
    predicted_train = models[i].predict(x_train)


    if (i == 1):
        print('Testing XGBRegressor():')
    else:
        print('Testing ' + str(models[i])+':')

    
    print('The mean absolute error between the training y values is: ' + str(mae(predicted_train, y_train )))
    print('The mean absolute error between the test y values is: ' + str(mae(predicted_test, y_test)))
    print()

    i = i + 1


