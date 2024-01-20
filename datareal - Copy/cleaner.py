import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("testsheet.xlsx")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def calculate_average(value):
    # Check if the value is a string
    if isinstance(value, str):
        # Use regex to find all numbers in the string
        numbers = re.findall(r"(\d+\.\d+|\d+)", value)
        # If there are two or more numbers
        if len(numbers) >= 2:
            # Calculate the average of the first two numbers
            return (float(numbers[0]) + float(numbers[1])) / 2
    # Return the original value if it's not a range
    return value


df["rent"] = df["rent"].combine_first(df["single-rent"])
df["rent"] = df["rent"].combine_first(df["rent2"])
df["walk-score"] = df["walk-score"].combine_first(df["walkability2"])
df["transit-score"] = df["transit-score"].combine_first(df["transit2"])
df["bike-score"] = df["bike-score"].combine_first(df["bike2"]).astype(float)
df["commute"] = df["commute"].combine_first(df["commute2"])
df["beds"] = df["beds"].combine_first(df["single-bedroom"])
df["baths"] = df["baths"].combine_first(df["single-bathrooms"])
df["schools"] = df["schools"].combine_first(df["schools2"])
df["footageplural"] = df["footageplural"].combine_first(df["footage-single"])
df["footageplural"] = df["footageplural"].str.strip().replace("--", np.nan)
df["footageplural"] = df["footageplural"].str.replace(",", "")
df["preview"] = df["overview2"].combine_first(df["preview"])
df["neighborhood"] = df["neighborhood"].combine_first(df["neighborhood-single"])

# df['footage'] = df['footage-single'].combine_first(df['footage-plural'])
df["factsfeatures"] = df["factsfeatures"].combine_first(df["ff2"])
df.drop(
    [
        "single-rent",
        "ff2",
        "rent2",
        "footage-single",
        "single-bathrooms",
        "single-bedroom",
        "commute2",
        "bike2",
        "transit2",
        "walkability2",
        "countbeds",
        "countbaths",
        "countrent2",
        "countfootageplural",
        "trueindex",
        "Unnamed: 0",
        "schools2",
        "overview2",
        "neighborhood-single",
        "page",
    ],
    axis=1,
    inplace=True,
)

# Apply the function to the column
df["rent"] = df["rent"].str.replace("$", "")
df["rent"] = df["rent"].str.replace("+", "")
df["rent"] = df["rent"].str.replace(",", "")
df["rent"] = df["rent"].apply(calculate_average)
df["rent"] = df["rent"].str.replace(r"\D+", "")
df["rent"] = df["rent"].str.replace(r"/mo", "")
df["rent"] = df["rent"].str.replace("--", "")
df = df[df["rent"] != ""]
df = df.dropna(subset=["rent"])
df["rent"] = df["rent"].astype(float)
df = df[df["rent"] < 20000]
df = df[df["rent"] > 0]
print(df["rent"].describe())


# Group by "page-href", "rent", "beds", and "baths"
df = (
    df.groupby(["page-href", "rent", "beds", "baths"], as_index=False)
    .first()
    .reset_index()
)
# Remove outliers
q1 = df['rent'].quantile(0.25)
q3 = df['rent'].quantile(0.75)
iqr = q3 - q1
df = df[~((df['rent'] < (q1 - 1.5 * iqr)) | (df['rent'] > (q3 + 1.5 * iqr)))]

print(df["rent"].describe())
# Define the range bins
bins = np.arange(0, df["rent"].max() + 200, 200)

# Create the histogram
plt.hist(df["rent"], bins=bins, edgecolor="black")

# Set the title and labels
plt.title("Frequency of Rent Ranges")
plt.xlabel("Rent")
plt.ylabel("Frequency")

# Show the plot
plt.show()

df.to_excel("output.xlsx")
