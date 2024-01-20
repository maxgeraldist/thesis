import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# from sklearn.model_selection import KFold
# Read the data
df = pd.read_excel("final_with_neighborhoods.xlsx")
# Split the data into training and testing sets
x = df.drop(["Index", "rent"], axis=1)
y = df["rent"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create a random forest classifier
rf = RandomForestRegressor(random_state=0)

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200, 300, 1000],
    "max_depth": [5, 8, 15, 25],
    "min_samples_split": [2, 5, 10, 15, 100],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["squared_error", "poisson", "friedman_mse", "absolute_error"],
}

# Define the k-fold
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=kfold, n_jobs=9, pre_dispatch=9, verbose=2
)


# # Create a random forest classifier with the best parameters
# rf_best = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=25,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     random_state=0,  # for reproducibility
# )

# Train the classifier
rf_best = grid_search.fit(x_train, y_train)

print("Best parameters found: ", grid_search.best_params_)

# Apply The Full Featured Classifier To The Test Data
with open("Random forest/output_with_neighborhood.txt", "w") as file:
    # Apply The Full Featured Classifier To The Test Data
    y_pred = rf_best.predict(x_test)

    # Calculate and write the MSE to the file
    mse = mean_squared_error(y_test, y_pred)
    file.write("Mean Squared Error: " + str(mse) + "\n")

    # Write the accuracy to the file
    accuracy = accuracy_score(y_test, y_pred)
    file.write("Accuracy: " + str(accuracy) + "\n")

    # Calculate and write the R² score to the file
    r2 = r2_score(y_test, y_pred)
    file.write("R² Score: " + str(r2) + "\n")

    # Write the classification report to the file
    report = classification_report(y_test, y_pred, zero_division=0)
    file.write("Classification Report:\n" + report + "\n")

    # Create a series with feature importances
    featimp = pd.Series(rf_best.feature_importances_, index=x.columns).sort_values(
        ascending=False
    )
    file.write("Feature Importances:\n" + str(featimp) + "\n")

    # Select 10 random rows from the dataset
    random_rows = df.sample(n=10, random_state=0)

    # Predict the rent's value for the selected rows
    random_rows_x = random_rows.drop(["Index", "rent"], axis=1)
    random_rows_y = random_rows["rent"]
    random_rows_y_pred = rf_best.predict(random_rows_x)

# Plot the predicted vs actual values for all rows
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Labels")
plt.ylabel("Predicted Labels")
plt.show()
