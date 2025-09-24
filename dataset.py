# Download data
data = pd.read_csv("endangered_languages.csv")
df = data
data.head()

# Data cleaning

df_clean = pd.read_csv("endangered_languages.csv")
print("Initial values:")
print(df_clean["Degree of endangerment"].value_counts())

# Clean the column
df_clean["Degree of endangerment"] = df_clean["Degree of endangerment"].astype(str).str.strip()

# Endangerment mapping
endangerment_mapping = {
    'Vulnerable': 0,
    'Definitely endangered': 1,
    'Severely endangered': 2,
    'Critically endangered': 3,
    'Extinct': 4
}

# Create the label encoded column
df_clean["Endangerment_encoded"] = df_clean["Degree of endangerment"].map(endangerment_mapping)
print("\nAfter mapping:")
print(df_clean[["Degree of endangerment", "Endangerment_encoded"]].head(10))
print(df_clean["Degree of endangerment"].value_counts())

# Plotting the graph
plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, 
              x="Degree of endangerment", 
              order=sorted(endangerment_mapping.keys(), key=lambda x: endangerment_mapping[x]))
plt.title("Distribution of Endangerment Levels")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
