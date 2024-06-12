import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#create fake data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

#split the dataset into training and testing bunches
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 56)

model = LinearRegression()

#training time!
model.fit(X_train, y_train)

#making predictions
y_pred = model.predict(X_test)

#evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Square Error: {mse}")

#printing the model's inercepts and coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")

#visualization
# Plot the results
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=2, label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Model')
plt.legend()
plt.show()

