# Predict Number of Endangered Languages per Country
def country_to_continent(country):
    try:
        country_code = pc.country_name_to_country_alpha2(country)
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        return pc.convert_continent_code_to_continent_name(continent_code)
    except:
        return None

# Apply to your Countries column
df_clean["Continent"] = df_clean["Countries"].apply(country_to_continent)

# Drop rows where continent couldn't be found
df_clean = df_clean.dropna(subset=["Continent"])

print(df_clean.head())

# Random Forest Regression
# Encode and Train a Regressor
# Label encode countries
country_counts = df_clean
target_column = "Endangerment_encoded"  

# Label encode countries if not already done
le_country = LabelEncoder()
country_counts["Countries_encoded"] = le_country.fit_transform(country_counts["Countries"])

# Features and target
X_reg = country_counts[["Number of speakers", "Latitude", "Longitude", "Countries_encoded"]]
y_reg = country_counts[target_column ]

# Split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Model
regressor = RandomForestRegressor(n_estimators=150, random_state=42)
regressor.fit(X_train_r, y_train_r)
y_pred_r = regressor.predict(X_test_r)

# Evaluation
print("MSE:", mean_squared_error(y_test_r, y_pred_r))
print("R2 Score:", r2_score(y_test_r, y_pred_r))

# Visualize Results
plt.figure(figsize=(8, 4))
plt.scatter(y_test_r, y_pred_r)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.title("Predicted vs Actual Endangered Language Count (per Country)")
plt.tight_layout()
plt.show()

# In the real work, I added Additional feature which is Linguistic Diversity Index (LDI) 
# Features used for retraining: "Number of speakers", "Latitude", "Longitude", "Countries_encoded", and "LDI"

# Expand Countries into Rows
# ------------------------------------------
df_reg = df_clean
df_expanded = df_reg.copy()

# Split comma-separated countries and explode into rows
df_expanded["Countries"] = df_expanded["Countries"].str.split(",\s*")
df_expanded = df_expanded.explode("Countries")
df_expanded["Countries"] = df_expanded["Countries"].str.strip().str.title()

# ------------------------------------------
# Load LDI & Merge
# ------------------------------------------
ldi_df = pd.read_csv("Linguistic_diversity_index.csv")  # Your LDI file
ldi_df["Country"] = ldi_df["Country"].str.strip().str.title()

# Merge LDI into exploded dataframe
df_expanded = df_expanded.merge(ldi_df, left_on="Countries", right_on="Country", how="left")

# Ensure LDI is numeric
df_expanded["LDI"] = pd.to_numeric(df_expanded["LDI"], errors="coerce")

# Optional: check how many are missing
print("Missing LDI values:", df_expanded["LDI"].isna().sum())

# ------------------------------------------
# Aggregate by Country
# ------------------------------------------
country_counts_ldi = df_expanded.groupby("Countries").agg({
    "ID": "count",
    "Number of speakers": "mean",
    "Latitude": "mean",
    "Longitude": "mean",
    "LDI": "mean"
}).rename(columns={"ID": "endangered_language_count"}).reset_index()

# Drop rows with missing LDI
country_counts_ldi = country_counts_ldi.dropna(subset=["LDI"])

# Label encode countries
le_country = LabelEncoder()
country_counts_ldi["Countries_encoded"] = le_country.fit_transform(country_counts_ldi["Countries"])

# ------------------------------------------
# Train Regression Model
# ------------------------------------------
# Define features and target
X_reg = country_counts_ldi[[
    "Number of speakers", "Latitude", "Longitude", "Countries_encoded", "LDI"
]]
y_reg = country_counts_ldi["endangered_language_count"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Model
regressor = RandomForestRegressor(n_estimators=150, random_state=42)
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ------------------------------------------
# Visualize Predictions
# ------------------------------------------
plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Endangered Language Count")
plt.ylabel("Predicted Count")
plt.title("Predicted vs Actual Endangered Language Count (per Country)")
plt.tight_layout()
plt.show()

# XGBoost Regression
# Define features and target

features = ["Number of speakers", "Latitude", "Longitude", "LDI"]
target = "endangered_language_count"

# Train-test split
X = country_counts_ldi[features]
y = country_counts_ldi[target]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train the XGBoost regressor
xgb_reg = XGBRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)
xgb_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# I also did XGBoost Regression Hyperparameter Tuning (with GridSearchCV) which is omitted here

# CatBoost for Regression
# Initialize the model
cat_model = CatBoostRegressor(
    iterations=500,       # Number of boosting rounds
    learning_rate=0.05,   # Learning rate
    depth=6,              # Tree depth
    loss_function='RMSE', # Loss function
    random_seed=42,
    verbose=50            # Shows training progress every 50 iterations
)

# Fit the model
cat_model.fit(X_train, y_train)

# Predict
y_pred = cat_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"CatBoost R² Score: {r2:.2f}")
print(f"CatBoost MSE: {mse:.2f}")

# Plot: Predicted vs Actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Endangered Language Count")
plt.ylabel("Predicted Count")
plt.title(f"CatBoost Regression: Predicted vs Actual\nR² Score = {r2:.2f}, MSE = {mse:.2f}")
plt.tight_layout()
plt.show()

# I blended CatBoost + Random Forest to enhance the capacity of the regression model
# Omitted here

# Continent-Level Forecast
continent_counts = df_clean.groupby("Continent").agg({
    "Name in English": "count",                   
    "Number of speakers": "mean",
    "Latitude": "mean",
    "Longitude": "mean"
}).rename(columns={"Name in English": "endangered_language_count"}).reset_index()

print(continent_counts)
