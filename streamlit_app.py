import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

url = 'https://raw.githubusercontent.com/OliverHu726/ML_in_FRE_HW6_app/main/regression_dataset.txt'
df = pd.read_csv(url, delimiter='\s+', header=None, names=['X', 'Y'])
Xs = df['X']
Ys = df['Y']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
  Xs, 
  Ys, 
  test_size=0.2, 
  random_state=42)

# Convert 1-d array to 2-d array so Linear Regression Model can process
X_train_2d = X_train.values.reshape(-1, 1)
X_test_2d = X_test.values.reshape(-1, 1)

# Fit model
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train_2d, y_train)
model = reg

# Assuming X_train_2d and Y_train are available from the training process
linear_reg_model.fit(X_train_2d, Y_train)

st.title('Linear Regression Prediction App')

# User input for the prediction
user_input = st.number_input('Enter a number:', value=0.0)

# Reshape the input to match the model's expectations
user_input_2d = np.array(user_input).reshape(1, -1)

# Make a prediction
prediction_result = linear_reg_model.predict(user_input_2d)[0]

# Display the prediction result
st.write(f'Prediction Result: {prediction_result:.4f}')
