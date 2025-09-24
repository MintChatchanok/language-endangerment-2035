# Map Country to Continent
import pycountry_convert as pc

def country_to_continent(country):
    try:
        country_code = pc.country_name_to_country_alpha2(country)
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        return pc.convert_continent_code_to_continent_name(continent_code)
    except:
        return None

# Apply mapping
df_clean["Continent"] = df_clean["Countries"].apply(country_to_continent)

# Drop rows with missing continents
df_clean = df_clean.dropna(subset=["Continent"])

# Encode Countries and Continents
from sklearn.preprocessing import LabelEncoder

# Encode country
if "Countries_encoded" not in df_clean.columns:
    le_country = LabelEncoder()
    df_clean["Countries_encoded"] = le_country.fit_transform(df_clean["Countries"])

# Encode continent
if "Continent_encoded" not in df_clean.columns:
    le_continent = LabelEncoder()
    df_clean["Continent_encoded"] = le_continent.fit_transform(df_clean["Continent"])

# Set Features and Target
features = [
    "Number of speakers",
    "Latitude",
    "Longitude",
    "Countries_encoded",
    "Continent_encoded"
]

target = "Endangerment_encoded"

X = df_clean[features]
y = df_clean[target]

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
