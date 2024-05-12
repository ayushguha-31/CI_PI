import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.title('Confidence Interval and Prediction Interval in Regression')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Adjustable parameters
n_samples = st.sidebar.slider('Number of Samples', min_value=50, max_value=500, value=100, step=10)
noise = st.sidebar.slider('Noise Level', min_value=1, max_value=50, value=10, step=1)
confidence_level = st.sidebar.slider('Confidence Level', min_value=0.1, max_value=0.99, value=0.95, step=0.01)

# Generate sample data
X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate confidence interval
confidence_interval = model.score(X_train, y_train), model.score(X_test, y_test)
st.write(f"Confidence Interval: {confidence_interval}")

# Calculate prediction interval using statsmodels
X_train_with_constant = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_with_constant).fit()
prediction_interval = model_sm.get_prediction(sm.add_constant(X_test)).summary_frame(alpha=1 - confidence_level)

# Plot the results
st.subheader('Scatter plot of actual vs predicted values with Confidence and Prediction Intervals')
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')

# Confidence Interval
plt.fill_between(X_test.flatten(), 
                 model.predict(X_test) - np.sqrt(model_sm.mse_resid), 
                 model.predict(X_test) + np.sqrt(model_sm.mse_resid), 
                 color='orange', alpha=0.5, label=f'{confidence_level*100}% Confidence Interval')

# Prediction Interval
plt.fill_between(X_test.flatten(), 
                 prediction_interval['obs_ci_lower'], 
                 prediction_interval['obs_ci_upper'], 
                 color='lightgray', alpha=0.5, label=f'{confidence_level*100}% Prediction Interval')

plt.xlabel('X_test')
plt.ylabel('y')
plt.title('Scatter plot with Confidence and Prediction Intervals')
plt.legend()
st.pyplot()

# Display the coefficients
st.subheader('Model Coefficients:')
st.write(f'Intercept: {model.intercept_}')
st.write(f'Coefficient: {model.coef_}')
