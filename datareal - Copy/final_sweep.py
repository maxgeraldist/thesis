import pandas as pd

df = pd.read_excel("parsed_with_neighborhood.xlsx", sheet_name="Sheet1")
df = df.drop(
    columns=[
        "index",
        "page-href",
        "web-scraper-order",
        "web-scraper-start-url",
        "factsfeatures",
        "commute",
        "schools",
        "preview",
        "policies2",
        "neighborhood",
        "address",
        "PD",
        "heating",
        "laundry",
        "flooring",
    ],
)

df["Index"] = df["Unnamed: 0"]
df = df.drop(columns=["Unnamed: 0"])
df.to_excel("final_with_neighborhoods.xlsx", sheet_name="Sheet1", index=False)
