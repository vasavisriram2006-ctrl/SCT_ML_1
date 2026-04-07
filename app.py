import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Load dataset
data = pd.read_csv("train.csv")
# Select features
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]
data = data.dropna()
# Visualization
plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
plt.scatter(data['GrLivArea'], data['SalePrice'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()
# Split data
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
# Save model
joblib.dump(model, "house_model.pkl")
# New prediction
new_house = [[2000, 3, 2]]
print("Predicted Price:", model.predict(new_house)[0])
