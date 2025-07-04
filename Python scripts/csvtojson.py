import pandas as pd
import json

# Load CSV file
csv_file = "water_quality_of_river_godavari-2014.csv"
df = pd.read_csv(csv_file, encoding="ISO-8859-1")

# Replace NaN values with None (which converts to null in JSON)
df = df.where(pd.notna(df), None)

# Convert to JSON
json_data = df.to_dict(orient="records")

# Save JSON file
json_file = "water_quality.json"
with open(json_file, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"JSON file saved as {json_file}")
