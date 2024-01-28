import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the Excel files
df1 = pd.read_excel("features_neighborhoods.xlsx")
df2 = pd.read_excel("features_no_neighborhoods.xlsx")

# Strip leading and trailing spaces from 'labels' column
df1["labels"] = df1["labels"].str.strip()
df1.sort_values("importances", inplace=True, ascending=False)
df1 = df1.head(10)  # Get the top 10 features
df2["labels"] = df2["labels"].str.strip()
df2.sort_values("importances", inplace=True, ascending=False)
df2 = df2.head(10)  # Get the top 10 features

# Merge the two dataframes on the feature names using an inner merge
df = pd.merge(df1, df2, on="labels", how="inner")

print(df)

# Sort the dataframe by the importances from the first model
df.sort_values("importances_x", ascending=False, inplace=True)

# Get the feature names and importances
features = df["labels"].tolist()
importances1 = df["importances_x"].tolist()
importances2 = df["importances_y"].tolist()

x = np.arange(len(features))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 8))  # Set the figure size
rects1 = ax.bar(
    x - width / 2,
    importances1,
    width,
    label="Model with neighborhoods",
    color="#B2B2FF",
    edgecolor="#0000FF",
)
rects2 = ax.bar(
    x + width / 2,
    importances2,
    width,
    label="Model without neighborhoods",
    color="#FFB2B2",
    edgecolor="#FF0000",
)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Importance")
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=90)  # Rotate labels for better readability
ax.legend()

fig.tight_layout()

plt.savefig("figure.png", dpi=100)  # Save the figure with a resolution of 125 dpi
plt.show()
