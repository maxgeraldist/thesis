import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_excel("final.xlsx")

if df.isnull().values.any():
    print("Warning: The DataFrame contains NaN values.")
else:
    print("The DataFrame does not contain any NaN values.")

# Define features and target
features = df.drop(["rent", "Index"], axis=1)
target = df["rent"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=0
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    "hidden_layer_sizes": [
        (50),
        (100),
        (200),
        (500),
        (50, 50),
        (100, 100),
        (200, 200),
        (500, 500),
        (50, 50, 50),
        (100, 100, 100),
        (200, 200, 200),
        (500, 500, 500),
        (50, 50, 50, 50),
        (100, 100, 100, 100),
        (200, 200, 200, 200),
        (500, 500, 500, 500),
    ],
    # "beta_1": [0.9, 0.8, 0.7],
    # "beta_2": [0.999, 0.99, 0.9],
    # "epsilon": [1e-08, 1e-07, 1e-06],
    # "activation": ["relu", "tanh"],
    "solver": ["adam", "lbfgs", "sgd"],
    "alpha": [0.0001, 0.001, 0.01],
    # "learning_rate": ["constant", "adaptive"],
    # "learning_rate_init": [0.001, 0.01, 0.1],
    "n_iter_no_change": [10],
}

# Create a MLPRegressor object
mlp = MLPRegressor(random_state=0, max_iter=100000)

# Create a KFold object with shuffling
kf = KFold(n_splits=3, shuffle=True, random_state=0)

# Create a GridSearchCV object
grid_search = GridSearchCV(
    mlp,
    param_grid,
    cv=kf,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2,
    pre_dispatch=24,
)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Get the best model
best_model = grid_search.best_estimator_

# Now you can use best_model to make predictions
y_pred = best_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")
