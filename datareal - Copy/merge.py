import pandas as pd
import glob
import os

# Define the directory to search for Excel files
directory = os.getcwd()

# Initialize the list to store dataframes
dfs = []

# Get a list of all excel files in the current directory
excel_files = glob.glob(os.path.join(directory, "raw_data", "*.xlsx"))

# Loop through the list of excel files and append each file's dataframe to the list
for file in excel_files:
    df = pd.read_excel(file)
    dfs.append(df)

# Concatenate all dataframes in the list
df = pd.concat(dfs, ignore_index=True)

df["web-scraper-order"] = df["web-scraper-order"].astype(str)
df["web-scraper-order"] = df["web-scraper-order"].str.replace("-", "")
df["rent2"] = df["rent"].combine_first(df["rent2"])


# Define a function to increment the counter
def increment_counter(df_page, column, counter):
    df_page = df_page.copy()  # Add this line
    counter[column] = 1
    for i in df_page.index:
        if pd.notnull(df_page.loc[i, column]):
            df_page.loc[i, "count" + column] = counter[column]
            counter[column] += 1
    return df_page


# Initialize counters
counter = {"beds": 1, "baths": 1, "footageplural": 1, "rent2": 1}

# Initialize count columns
for column in counter.keys():
    df["count" + column] = None

# Get the unique 'page-href' values
page_hrefs = df["page-href"].unique()

for page_href in page_hrefs:
    # Filter the dataframe for the current 'page-href'
    df_page = df[df["page-href"] == page_href]

    # Apply the function to each df_page
    df_page = increment_counter(df_page, "beds", counter)
    df_page = increment_counter(df_page, "baths", counter)
    df_page = increment_counter(df_page, "footageplural", counter)
    df_page = increment_counter(df_page, "rent2", counter)

    # Update the rows in df
    df.update(df_page)

# Create the 'trueindex' column
for column in counter.keys():
    df.loc[df["count" + column].notnull(), "trueindex"] = (
        df["count" + column].astype(str) + df["page-href"]
    )

# Group by 'trueindex', take the first row of each group, and sort by 'web-scraper-order'
df1 = (
    df[df["trueindex"].notnull()]
    .groupby("trueindex")
    .first()
    .sort_values("web-scraper-order")
)

# Concatenate with the rows where 'trueindex' is null
df2 = pd.concat([df1, df[df["trueindex"].isnull()]])

# Concatenate with the rows where 'beds', 'baths', 'rent2', and 'footageplural' are all null
df2 = pd.concat(
    [df2, df[df[["beds", "baths", "rent2", "footageplural"]].isnull().all(axis=1)]]
)

df2.to_excel("testsheet.xlsx")
