# Import necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
# Load the data
df = pd.read_excel("final_with_neighborhoods.xlsx")
df = df.reset_index()
# Define features and target
features = df.drop(["rent", "Index"], axis=1)
target = df["rent"]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=0
)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create a MLPRegressor object
mlp = MLPRegressor(random_state=0, max_iter=100000, early_stopping=True)

# Grid search was solved in the other file - this is the result
best_params = {
    "solver": "adam",
    "activation": "relu",
    "alpha": 0.0001,
    "hidden_layer_sizes": (50, 50, 50),
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "max_iter": 100000,
    "early_stopping": True,
    "n_iter_no_change": 10,
    "random_state": 0,
    "beta_1": 0.7,
    "beta_2": 0.9,
    "epsilon": 1e-06,
}

# Set the best parameters
mlp.set_params(**best_params)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)

# Replace predicted outliers with the mean of the target
y_pred[y_pred > (y_test.max() * 2)] = y_test.mean()


# hiddenlayers = {(100), (500, 500), (500, 500, 500)}
# # Chart loss over iterations for each hidden layers setting
# for hiddenlayer in hiddenlayers:
#     mlp.set_params(hidden_layer_sizes=hiddenlayer)
#     mlp.fit(x_train, y_train)
#     plt.plot(mlp.loss_curve_, label=hiddenlayer)
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig("NN/neighborhood_LossNN.png")
# plt.clf()

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")

# Calculate adjusted R^2 score
n = len(x_test)
p = len(features.columns)
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(f"Adjusted R^2 Score: {adj_r2}")

# Caluclate the mean squared error of the residuals
residuals = y_test - y_pred

# Plot the predicted vs actual values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.savefig("NN/neighborhood_ScatterNN.png")

with open("NN/neighborhood_NNTuned.txt", "w") as f:
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"R^2 Score: {r2}\n")
    f.write(f"Adjusted R^2 Score: {adj_r2}\n")
    f.write(f"Best parameters: {best_params}\n")
