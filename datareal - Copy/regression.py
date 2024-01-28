import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
plt.figure(figsize=(10, 8))

# Read the data
df = pd.read_excel("final.xlsx")

# assign y and x, drop variables with high VIF
y = df["rent"]
x = df.drop(
    [
        "rent",
        "Index",
        "pets_deposit",
        "Commute_nonrush",
        "Commute_rush",
        "school3",
        "walk-score",
        "bike-score",
        "lease_term",
        "transit-score",
        "cats_allowed",
        "pets_allowed",
        "deposit_missing",
        "application_fee_missing",
        "application_fee",
        "school2",
        "garbageincluded",
        "pets_allowed_deposit",
        "arrests",
        # "Lake View",
        # "East Ukrainian Village",
        # "View",
        # "Near North",
        # "South Loop",
        # "Rogers Park",
        # "Gold Coast",
        # "Hyde Park",
        # "Unknown",
    ],
    axis=1,
)

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
with open("OLS/vif.txt", "w") as f:
    f.write(str(vif))

matrix = x.corr()
with open("OLS/matrix.txt", "w") as f:
    f.write(str(matrix))

# Split the data into training and testing sets
x_1, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Drop the exogenous variables
x_1 = sm.add_constant(
    x_1.drop(["deposit", "carpet", "is_studio", "pets_rent"], axis=1)
).reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Fit the model
model = sm.OLS(y_train, x_1).fit()
model = model.get_robustcov_results(cov_type="HC3")

# Calculate residuals
residuals = model.resid
residuals = pd.Series(residuals).reset_index(drop=True)

# Perform Breusch-Pagan test
x_1_with_constant = sm.add_constant(x_1)
model_with_constant = sm.OLS(y_train, x_1_with_constant).fit()
residuals_with_const = model_with_constant.resid
bp_test = het_breuschpagan(residuals_with_const, x_1_with_constant)
print(f"Breusch-Pagan test statistic: {bp_test[0]}")
print(f"Breusch-Pagan test p-value: {bp_test[1]}")

t_test = stats.ttest_1samp(residuals, 0)
print(f"One-sample t-test statistic: {t_test.statistic}")
print(f"One-sample t-test p-value: {t_test.pvalue}")

# Perform Shapiro-Wilk test on residuals
shapiro_test = stats.shapiro(residuals)
print(f"Shapiro-Wilk test statistic: {shapiro_test.statistic}")
print(f"Shapiro-Wilk test p-value: {shapiro_test.pvalue}")

# Exogeneity test
# Fit a model with residuals as the dependent variable and the original X
# as the independent variables
model_exog = sm.OLS(residuals, sm.add_constant(x_1)).fit()
print(model_exog.summary())

# Subset the test data to include only the selected features
x_test_BE = sm.add_constant(x_test)[list(x_1.columns)]

summary_str = model.summary().as_text()
with open("OLS/summary_OLS.txt", "w") as f:
    f.write(summary_str)

# Make predictions on the test data
y_pred = model.predict(x_test_BE)


# Plot predicted vs actual
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.savefig("OLS/predicted_vs_actual.png")
plt.clf()

# Plot residuals vs fitted
plt.scatter(pd.Series(model.fittedvalues), pd.Series(model.resid), alpha=0.5)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.savefig("OLS/residuals_vs_fitted.png")

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
with open("OLS/mse.txt", "w") as f:
    f.write(str(mse))

n = len(y_test)  # number of observations
p = x_test.shape[1]  # number of predictors

# Calculate and print the Mean Squared Error of the Residuals (MSER)
mser = mse / (n - p - 1)
print("MSER:", mser)
with open("OLS/mser.txt", "w") as f:
    f.write(str(mser))

# Calculate and print the R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)
with open("OLS/r2.txt", "w") as f:
    f.write(str(r2))

# Calculate and print the Adjusted R-squared score
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R-squared:", adj_r2)
with open("OLS/adj_r2.txt", "w") as f:
    f.write(str(adj_r2))
