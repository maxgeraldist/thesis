import pandas as pd

# Load the Excel file
df = pd.read_excel("final.xlsx")

# List of columns to calculate statistics for
columns = [
    "rent",
    "beds",
    "baths",
    "walk-score",
    "transit-score",
    "bike-score",
    "footageplural",
    "Commute_rush",
    "Commute_nonrush",
    "arrests",
]

# Open the text file to write the statistics
with open("summary_statistics.txt", "w") as f:
    for column in columns:
        if column in df.columns:
            # Calculate statistics
            min_val = df[column].min()
            max_val = df[column].max()
            median_val = df[column].median()
            mean_val = df[column].mean()
            std_val = df[column].std()

            # Write statistics to the file
            f.write(f"Statistics for {column}:\n")
            f.write(f"Min: {min_val}\n")
            f.write(f"Max: {max_val}\n")
            f.write(f"Median: {median_val}\n")
            f.write(f"Mean: {mean_val}\n")
            f.write(f"Standard Deviation: {std_val}\n\n")
        else:
            f.write(f"{column} is not in the dataframe.\n\n")
