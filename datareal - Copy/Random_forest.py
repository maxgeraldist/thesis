import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Read the data
df = pd.read_excel("final_with_neighborhoods.xlsx")
plt.figure(figsize=(10, 8))

# Split the data into training and testing sets
x = df.drop(["Index", "rent"], axis=1)
y = df["rent"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Create a random forest classifier with the best parameters - without and without neighborhood (it's the same)
rf_best = RandomForestRegressor(
    n_estimators=1000,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion="absolute_error",
    random_state=0,
)

# Train the classifier
rf_best.fit(x_train, y_train)

# Print the best hyperparameters

# Predict the labels of the test set
y_pred = rf_best.predict(x_test)

n = x_test.shape[0]  # number of samples
p = x_test.shape[1]  # number of predictors
r2 = r2_score(y_test, y_pred)


feature_number = 10
featimp = pd.Series(rf_best.feature_importances_, index=x.columns).sort_values(
    ascending=False
)
# Slice the series to only include the first feature_number features
featemp = featimp[:feature_number]
featemp.plot(kind="barh", title="Feature Importances")
plt.ylabel("Feature Importance Score")
plt.savefig("Random Forest/Feature_importance_RF_with_neighborhood.png")
plt.clf()

# Apply The Full Featured Classifier To The Test Data
y_pred = rf_best.predict(x_test)

# Calculate residuals
residuals = y_test - y_pred

# Apply The Full Featured Classifier To The Test Data
with open("Random Forest/output_with_neighborhood.txt", "w") as file:
    # Calculate and write the MSE to the file
    mse = np.mean(residuals**2)
    file.write("Mean Squared Error: " + str(mse) + "\n")

    # Calculate and write the MSE of the residuals to the file
    residuals = y_test - y_pred
    mse_residuals = mean_squared_error(y_test, residuals)
    file.write("Mean Squared Error of the residuals: " + str(mse_residuals) + "\n")

    # Calculate and write the R² score to the file
    r2 = r2_score(y_test, y_pred)
    file.write("R² Score: " + str(r2) + "\n")

    # Calculate and write the Adjusted R² score to the file
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    file.write("Adjusted R² Score: " + str(adjusted_r2) + "\n")

    # Create a series with feature importances
    file.write("Feature Importances:\n" + str(featimp) + "\n")

    # Write down the best parameters
    params = rf_best.get_params()
    file.write("Best parameters: " + str(params) + "\n")

# Plot the predicted vs actual values for all rows
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.savefig("Random Forest/Predicted_vs_actual_RF_with_neighborhood.png")
