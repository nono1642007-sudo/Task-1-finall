import pandas as pd
df = pd.read_csv("House_Prices.csv")
print("Original Dataset:")
print(df.head())
print("Dataset Shape:")
print(df.shape)
print("Missing Values Before Processing:")
print(df.isnull().sum())
numerical_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(include=["object", "category", "string"]).columns
for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].mode()[0])
print("Missing Values After Processing:")
print(df.isnull().sum())

print("Duplicate Rows Before Processing:")
print(df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate Rows After Processing:")
print(df.duplicated().sum())

categorical_columns = df.select_dtypes(include=["object", "category", "string"]).columns
print("Categorical Features:")
print(categorical_columns.tolist())

high_cardinality_columns = []
low_cardinality_columns = []

for column in categorical_columns:
    unique_values = df[column].nunique()
    if unique_values > 20:
        high_cardinality_columns.append(column)
    else:
        low_cardinality_columns.append(column)
        
print("High Cardinality Columns:")
print(high_cardinality_columns)
print("Low Cardinality Columns:")
print(low_cardinality_columns)

for column in high_cardinality_columns:
    frequency = df[column].value_counts(normalize=True)
    df[column] =df[column].map(frequency)

df = pd.get_dummies(df,
    columns=low_cardinality_columns,
    drop_first=True,
    dtype=int

)
print("Final Dataset:")
print(df.head())
print("Final Dataset Shape:")
print(df.shape)
print("Final Columns:")
print(df.columns.tolist())
print("Final Missing Values:")
print(df.isnull().sum().sum())

df.to_csv(
    "house_prices_cleaned.csv",
    index=False
)
print("Data Preprocessing Completed Successfully!")
print("Cleaned dataset saved as:")
print("house_prices_cleaned.csv")