import pandas as pd

INPUT_CSV = 'cardio_train_preprocessed_full.csv'

# Load the preprocessed data
df = pd.read_csv(INPUT_CSV)

# Check if all rows with Alcohol==1 also have High Blood Pressure==1
mask = (df['High Blood Pressure'] == 1) & (df['Overweight'] == 1) & (df['HighGlucose'] == 1) & (df['Smoke'] == 1) & (df['HighCholesterol'] == 1)
violating_rows = df[mask & (df['Alcohol'] == 0)]

if not violating_rows.empty:
    print("Found a row where Alcohol==1 and High Blood Pressure==0:")
    print(violating_rows.iloc[0])
else:
    print("All rows with Alcohol==1 also have High Blood Pressure==1.") 