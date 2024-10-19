This project aims to predict the number of calories burnt during physical activities by using physical characteristics such as age, gender, height, weight, heart rate, etc. Inspired by GeeksforGeeks "Calories Burnt Prediction using Machine Learning," this project serves as a learning tool to create and assess the accuracy of various machine learning algorithms and models. This is an evolving project. 

DATASETS:
The dataset used for this project is a combination of two datasets "calories.csv" and "exercise.csv," both of which can be found in this repository. 

The combined dataset includes the following data points relevant to individuals' physical features:
- Gender (Male/Female)
- Age (years)
- Height (cm)
- Weight (kg)
- Heart rate (bpm)
- Duration of activity (minutes)
- Calories burned

Note, this data was not verified and should not be considered as accurate. 

MACHINE LEARNING MODELS:
The following Python machine learning models were used and compared on the same dataset.
- LinearRegression()
- XGBRegressor()
- Lasso()
- RandomForestRegressor()
- Ridge()

RESULTS:
Comparing results utilizes the mean absolute error metric (MAE). The MAE is calculated for training and testing data and then compared.

NEXT STEPS:
- Enhance the data pre-processing procedure
- Explore more machine learning models
- Use additional metrics to assess prediction accuracy (mean squared error, mean absolute percentage error, etc.)
- Visualize/tabulate burnt calories predictions
