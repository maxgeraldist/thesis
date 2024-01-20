# Import necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
df = pd.read_excel("final.xlsx")
df = df.reset_index()
maxrent = df["rent"].max()
meanrent = df["rent"].mean()

# Define features and target
features = df.drop(["rent", "Index"], axis=1)
target = df["rent"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=0
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a MLPRegressor object
mlp = MLPRegressor(random_state=0, max_iter=100000, early_stopping=True)

# Grid search was solved in the other file - this is the result
best_params = {
    "activation": "relu",
    "alpha": 0.0001,
    "hidden_layer_sizes": (500, 500),
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "solver": "adam",
    "early_stopping": True,
    "beta_1": 0.7,
    "beta_2": 0.999,
    "epsilon": 1e-07,
    "max_iter": 100000,
}

# Set the best parameters
mlp.set_params(**best_params)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

countreplace = 0
for value in y_pred:
    if value > maxrent:
        y_pred[y_pred == value] = meanrent
        countreplace += 1


# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Number of replaced values: {countreplace}")

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter plot of Predicted vs Actual Values")
plt.show()

with open("NN/NNTuned.txt", "w") as f:
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"R^2 Score: {r2}\n")
    f.write(f"Best parameters: {best_params}\n")

solvers = ["lbfgs", "sgd", "adam"]

# Initialize the MLPRegressor
mlp = MLPRegressor(**best_params)

# Fit the models and track the loss at each iteration
for solver in solvers:
    mlp.set_params(hidden_layer_sizes=(500, 500), solver=solver)
    mlp.fit(X_train, y_train)
    losses = mlp.loss_curve_
    plt.plot(losses, label=f"{solver}")

# Plot the loss curves
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Define the number of neurons for the two-layer model
neurons = [(50, 50), (100, 100), (200, 200), (500, 500)]

# Initialize lists to store the results
neuron_counts = []
accuracies = []

# Run the regressions and track the accuracy
for n in neurons:
    mlp.set_params(hidden_layer_sizes=(n,))
    mlp.fit(X_train, y_train)
    accuracy = mlp.score(X_test, y_test)
    neuron_counts.append(n)
    accuracies.append(accuracy)

# Plot the results
plt.plot(neuron_counts, accuracies, "o-")
plt.xlabel("Number of Neurons")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Number of Neurons in Single-Layer Regression")
plt.grid(True)
plt.savefig("accuracy_vs_neurons.png")
plt.show()
