import numpy as np
import pandas as pd
import re
from district_dict import neighborhood_to_pd, district_to_crime

df: pd.DataFrame = pd.read_excel("output.xlsx", sheet_name="Sheet1")
df["commute"] = df["commute"].replace("^$", np.nan, regex=True)
df["Commute_rush"] = pd.to_numeric(
    df["commute"].str.split(r"\D+", expand=True)[3], errors="coerce"
)
df["Commute_nonrush"] = pd.to_numeric(
    df["commute"].str.split(r"\D+", expand=True)[4], errors="coerce"
)

df["footageunknown"] = 0


def first_non_null(series):
    non_nulls = series.dropna()
    if len(non_nulls) > 0:
        first_non_null_value = non_nulls.iloc[0]
        return series.fillna(first_non_null_value)
    else:
        return series


# Apply the function to the 'address' column within each group
df["address"] = df["address"].combine_first(df["address-single"])
df.drop("address-single", axis=1, inplace=True)
df["address"] = df.groupby("page-href")["address"].transform(first_non_null)
df["neighborhood"] = df.groupby("page-href")["neighborhood"].transform(first_non_null)
df["Commute_rush"] = df.groupby("page-href")["Commute_rush"].transform(first_non_null)
df["Commute_nonrush"] = df.groupby("page-href")["Commute_nonrush"].transform(
    first_non_null
)
df["index"] = df["address"].str.extract(r"(\d{5})\s*$")
df["neighborhood"] = df.groupby("index")["neighborhood"].transform(first_non_null)
df["index"] = df.groupby("neighborhood")["index"].transform(first_non_null)
df["Commute_rush"] = (
    df["Commute_rush"]
    .fillna(df.groupby("index")["Commute_rush"].transform("mean"))
    .fillna(df["Commute_rush"].mean())
)  # imputing
df["Commute_nonrush"] = (
    df["Commute_nonrush"]
    .fillna(df.groupby("index")["Commute_nonrush"].transform("mean"))
    .fillna(df["Commute_nonrush"].mean())
)  # imputing
df["Commute_rush"] = (
    df["Commute_rush"]
    .fillna(df.groupby("neighborhood")["Commute_rush"].transform("mean"))
    .fillna(df["Commute_rush"].mean())
)  # imputing
df["Commute_nonrush"] = (
    df["Commute_nonrush"]
    .fillna(df.groupby("neighborhood")["Commute_nonrush"].transform("mean"))
    .fillna(df["Commute_nonrush"].mean())
)  # imputing
df["neighborhood"] = (
    df["neighborhood"]
    .replace("Neighborhood: |Building overview|Facts and features", "", regex=True)
    .str.strip()
    .fillna("Unknown")
    .replace("", "Unknown")
)
for neighborhood in df["neighborhood"].unique():
    df[neighborhood] = df["neighborhood"].str.contains(neighborhood).astype(int)

df["PD"] = df["neighborhood"].map(neighborhood_to_pd)
df["arrests"] = df["PD"].map(district_to_crime)
df["arrests"] = df["arrests"].fillna(1503)  # imputing by city mean


def convert_footage(row):
    try:
        row["footageplural"] = int(row["footageplural"])
    except:
        row["footageunknown"] = 1
        row["footageplural"] = np.mean(df["footageplural"].dropna())  # imputing
    return row


df = df.apply(convert_footage, axis=1)


def impute_schools(row):  # imputing
    schools = ["school1", "school2", "school3"]
    for school in schools:
        try:
            row[school] = int(row[school])
        except:
            # Create a list of scores that are not NaN
            scores_not_nan = [
                row[each_school]
                for each_school in schools
                if pd.notnull(row[each_school])
            ]
            # Calculate the mean of the scores that are not NaN
            row[school] = (
                sum(scores_not_nan) / len(scores_not_nan) if scores_not_nan else np.nan
            )
    return row


matches = df["schools"].str.extractall(r"(\d+)/").unstack()
df["school1"] = matches[0][0].astype(int)
df["school2"] = matches[0][1]
df["school3"] = matches[0][2]

df = df.apply(impute_schools, axis=1)

# Dictionary mapping lease terms to months
lease_terms_to_months = {
    "Six months": 6,
    "One year": 12,
    "Two years": 24,
    "Three months": 3,
    # Add more if needed
}
df["is_flexible"] = df["policies2"].str.contains("Flexible").fillna(False).astype(int)

# Remove 'Flexible' from the 'policies2' column
df["policies2"] = df["policies2"].str.replace("Flexible", "")

# Initialize 'lease_term' column with NaN
df["lease_term"] = -1

# Replace lease terms with their mapped values
for term, months in lease_terms_to_months.items():
    mask = df["policies2"].str.contains("Lease Terms" + term).fillna(False)
    df.loc[mask, "lease_term"] = months


def check_utilities(s: str, utility: str) -> bool:
    # Check if s is a string
    if isinstance(s, str):
        # Find the index of "Utilities included in rent" in the string
        uiir_index = s.find("Utilities included in rent")

        # If "Utilities included in rent" is not found, return False
        if uiir_index == -1:
            return False

        # Extract the part of the string after "Utilities included in rent"
        after_uiir = s[uiir_index + len("Utilities included in rent") :]

        # Check if utility is in the part of the string after "Utilities included in rent"
        return utility in after_uiir
    else:
        # If s is not a string, return False
        return False


# If 'lease_term' is still -1, try to extract a number that is not preceded by a dollar sign
mask = df["lease_term"] == -1
df.loc[mask, "lease_term"] = df.loc[mask, "policies2"].str.findall(
    r"(?<=Lease\sTerms)(\d+)"
)

df.loc[df["lease_term"] == -1, "lease_term"] = np.nan

df["lease_term"] = df["lease_term"].apply(
    lambda x: int(x[0]) if isinstance(x, list) and x else None
)

# Convert 'lease_term' to numeric
df["lease_term"] = pd.to_numeric(df["lease_term"], errors="coerce").fillna(
    12
)  # imputing, see https://www.bls.gov/spotlight/2022/housing-leases-in-the-u-s-rental-market/home.htm

df["application_fee"] = df["policies2"].str.extract("\$(\d+) application fee")
df["application_fee_missing"] = df["application_fee"].isnull().astype(int)
bins = np.arange(0, df["rent"].max() + 200, 200)
df["rent_bracket"] = pd.cut(df["rent"], bins, include_lowest=True)
df["application_fee"] = pd.to_numeric(df["application_fee"], errors="coerce")
df["rent_bracket"] = pd.to_numeric(df["rent_bracket"], errors="coerce")
# Group by the rent bracket instead of neighborhood
df["application_fee"] = (
    df["application_fee"]
    .fillna(df.groupby("rent_bracket")["application_fee"].transform("mean"))
    .fillna(df["application_fee"].mean())
    .astype(int)
)  # imputing per rent bracket, then by global mean
df["deposit"] = df["preview"].str.replace(",", "").str.extract("Deposit \& fees\$(\d+)")
df["deposit"] = pd.to_numeric(df["deposit"], errors="coerce")
df["deposit_missing"] = df["deposit"].isnull().astype(int)
df["deposit"] = (
    df["deposit"]
    .fillna(df.groupby("rent_bracket")["deposit"].transform("mean"))
    .fillna(df["deposit"].mean())
    .astype(int)
)  # imputing per rent bracket, then by global mean
df = df.drop(columns=["rent_bracket"])

df["electricityincluded"] = (
    df["policies2"].apply(check_utilities, utility="Electricity").astype(int)
)
df["waterincluded"] = (
    df["policies2"].apply(check_utilities, utility="Water").astype(int)
)
df["gasincluded"] = df["policies2"].apply(check_utilities, utility="Gas").astype(int)
df["garbageincluded"] = (
    df["policies2"].apply(check_utilities, utility="Garbage").astype(int)
)
df["sewerincluded"] = (
    df["policies2"].apply(check_utilities, utility="Sewer").astype(int)
)

df["internetincluded"] = (
    df["policies2"].apply(check_utilities, utility="Internet").astype(int)
)

df["cableincluded"] = (
    df["policies2"].apply(check_utilities, utility="Cable").astype(int)
)

df["heatincluded"] = df["policies2"].apply(check_utilities, utility="Heat").astype(int)

mask_large = df["policies2"].str.lower().str.contains("dogsallowedlarge").fillna(False)
mask_small = df["policies2"].str.lower().str.contains("dogsallowedsmall").fillna(False)
mask_cats = df["policies2"].str.lower().str.contains("catsallowed").fillna(False)

# Use the masks to assign the corresponding values
df.loc[mask_large, "large_dogs_allowed"] = 1
df["large_dogs_allowed"] = df["large_dogs_allowed"].fillna(0)
df.loc[mask_small & ~mask_large, "small_dogs_allowed"] = 1
df["small_dogs_allowed"] = df["small_dogs_allowed"].fillna(0)
df.loc[mask_cats & ~mask_large & ~mask_small, "cats_allowed"] = 1
df["cats_allowed"] = df["cats_allowed"].fillna(0)
df.loc[df["beds"] != "Studio", "is_studio"] = 0
df.loc[df["beds"] == "Studio", "is_studio"] = 1
df["beds"] = df["beds"].replace("Studio", 0).astype(int)
df["baths"] = df["baths"].replace("--", 0).astype(float)


df["hasgarage"] = (
    df["factsfeatures"].str.contains("Garage", case=False).fillna(0).astype(int)
)
df["hassurfaceparking"] = (
    df["factsfeatures"]
    .str.contains("Surface Parking", case=False)
    .fillna(0)
    .astype(int)
)
df["hasstreetparking"] = (
    df["factsfeatures"].str.contains("Street Parking", case=False).fillna(0).astype(int)
)


df["pets_deposit"] = (
    df["policies2"].str.replace(",", "").str.extract("\$(\d+) one time fee")
)
df["pets_deposit"] = pd.to_numeric(df["pets_deposit"], errors="coerce")
df["pets_deposit"] = df["pets_deposit"].fillna(df["pets_deposit"].mean()).astype(int)
df["pets_allowed_deposit"] = np.where(
    (
        (df["large_dogs_allowed"] == 1)
        | (df["small_dogs_allowed"] == 1)
        | (df["cats_allowed"] == 1)
    ),
    df["pets_deposit"],
    0,
)
df["pets_allowed"] = df["policies2"].str.extract("(\d+) pet max")
df["pets_allowed"] = (
    df["pets_allowed"]
    .fillna(
        (df[["large_dogs_allowed", "small_dogs_allowed", "cats_allowed"]] == 1).any(
            axis=1
        )
    )
    .astype(int)
)

df["pets_rent"] = df["policies2"].str.extract("\$(\d+) monthly pet fee")
df["pets_rent"] = pd.to_numeric(df["pets_rent"], errors="coerce")
df["pets_rent"] = df["pets_rent"].fillna(df["pets_rent"].mean()).astype(int)
df["pets_rent"] = np.where(
    (
        (df["large_dogs_allowed"] == 1)
        | (df["small_dogs_allowed"] == 1)
        | (df["cats_allowed"] == 1)
    ),
    df["pets_rent"],
    0,
)


df["heating"] = df["factsfeatures"].str.extract(
    "Heating.*?(Forced Air|Radiant|Baseboard|Electric|Gas|None)"
)

df["laundry"] = df["factsfeatures"].str.extract(
    "Laundry.*?(Shared Laundry|In Unit|in-unit|None)", flags=re.IGNORECASE
)


df["flooring"] = df["factsfeatures"].str.extract(
    "Flooring.*?(Hardwood|Carpet|Tile|Laminate|Linoleum)",
    flags=re.IGNORECASE,
)

df["hardwood"] = (
    df["flooring"].str.contains("Hardwood", flags=re.IGNORECASE).fillna(0).astype(int)
)
df["carpet"] = (
    df["flooring"].str.contains("Carpet", flags=re.IGNORECASE).fillna(0).astype(int)
)
df["tile"] = (
    df["flooring"].str.contains("Tile", flags=re.IGNORECASE).fillna(0).astype(int)
)
df["laminate"] = (
    df["flooring"].str.contains("Laminate", flags=re.IGNORECASE).fillna(0).astype(int)
)
df["linoleum"] = (
    df["flooring"].str.contains("Linoleum", flags=re.IGNORECASE).fillna(0).astype(int)
)

df["gated"] = (
    df["factsfeatures"].str.contains("Gated", case=False).fillna(0).astype(int)
)
df["hasftness"] = (
    df["factsfeatures"].str.contains("Fitness", case=False).fillna(0).astype(int)
)
df["haspool"] = (
    df["factsfeatures"].str.contains("Pool", case=False).fillna(0).astype(int)
)
df["hasac"] = (
    df["factsfeatures"]
    .str.contains("Air Conditioning", case=False)
    .fillna(0)
    .astype(int)
)
df["hasdishwasher"] = (
    df["factsfeatures"].str.contains("Dishwasher", case=False).fillna(0).astype(int)
)
df["hasfireplace"] = (
    df["factsfeatures"].str.contains("Fireplace", case=False).fillna(0).astype(int)
)
df["year_built"] = df["factsfeatures"].str.extract("Year built: (\d+)")
df["year_built"] = (
    df["year_built"].fillna(df["year_built"].notnull().mean()).astype(int)
)  # imputing


df["hasbalcony"] = (
    df["factsfeatures"].str.contains("Balcony", case=False).fillna(0).astype(int)
)

df["heating_electric"] = (
    df["heating"].str.contains("Electric", flags=re.IGNORECASE).fillna(0).astype(int)
)
df["heating_gas"] = (
    df["heating"].str.contains("Gas", flags=re.IGNORECASE).fillna(0).astype(int)
)
df["heating_forced_air"] = (
    df["heating"].str.contains("Forced Air", flags=re.IGNORECASE).fillna(0).astype(int)
)
df["heating_radiant"] = (
    df["heating"].str.contains("Radiant", flags=re.IGNORECASE).fillna(0).astype(int)
)
df["heating_baseboard"] = (
    df["heating"].str.contains("Baseboard", flags=re.IGNORECASE).fillna(0).astype(int)
)
df["laundry_shared"] = (
    df["laundry"].str.contains("Shared", flags=re.IGNORECASE).fillna(0).astype(int)
)
df["laundry_in_unit"] = (
    df["laundry"]
    .str.contains("In Unit|in-unit", flags=re.IGNORECASE)
    .fillna(0)
    .astype(int)
)

df["bike-score"] = df["bike-score"].fillna(int(df["bike-score"].mean()))  # imputing
df["transit-score"] = df["transit-score"].fillna(
    int(df["transit-score"].mean())
)  # imputing
df["walk-score"] = df["walk-score"].fillna(int(df["walk-score"].mean()))  # imputing


df.to_excel("parsed_with_neighborhood.xlsx", index=False)
