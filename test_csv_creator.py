import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch dataset
bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets
df = pd.concat([X, y], axis=1)

# Create small test file
test_data = df.sample(n=100, random_state=42)
test_data.to_csv('test_100.csv', index=False)
print("✅ Created test_100.csv")
print(f"Columns: {test_data.columns.tolist()}")
print(f"Shape: {test_data.shape}")
print(f"Target values: {test_data['y'].unique()}")