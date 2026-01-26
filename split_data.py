import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the original processed dataset
# Ensure 'Processed_Mental_Health_Data.csv' is in your project folder
try:
    df = pd.read_csv('Processed_Mental_Health_Data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The processed data file was not found.")

# 2. Split the data into Training (80%) and Testing (20%) sets
# random_state ensures reproducibility of the results
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Export the split data into separate CSV files for verification
train_df.to_csv('Training_Data_80.csv', index=False)
test_df.to_csv('Testing_Data_20.csv', index=False)

# 4. Print summary statistics for documentation
print("--- Data Splitting Summary ---")
print(f"Total samples in dataset: {len(df)}")
print(f"Training samples (80%): {len(train_df)}")
print(f"Testing samples (20%): {len(test_df)}")
print("Files created: 'Training_Data_80.csv' and 'Testing_Data_20.csv'")