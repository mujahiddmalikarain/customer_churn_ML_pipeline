import pandas as pd    #import pandas library to read excel file and manipulate data

path="../data/telco_churn.xlsx"   #data path to the excel file containing the telco churn data


df = pd.read_excel(path)   # read the excel file and store it in a pandas dataframe called df
print(df.head())

#check data shape
print(df.shape)


print(df.columns.tolist())   #print the column names of the dataframe as a list

print(df.head(3))   #print the first 3 rows of the dataframe to check the data

print("\nMissing values (top 10):\n", df.isna().sum().sort_values(ascending=False).head(10))


# Target column check (common name in Telco dataset)
possible_targets = ["Churn", "churn", "Exited", "target", "Target"]
found = [c for c in df.columns if c in possible_targets]
print("\nPossible target columns found:", found)