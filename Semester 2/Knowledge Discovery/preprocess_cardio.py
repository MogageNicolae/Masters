import pandas as pd

INPUT_CSV = 'cardio_train.csv'
OUTPUT_CSV = 'cardio_train_triadic.csv'

# Read the data
# Only load necessary columns for efficiency
cols_needed = ['id', 'ap_hi', 'ap_lo', 'height', 'weight']
df = pd.read_csv(INPUT_CSV, sep=';')

df = df[df['cardio'] == 1]

# Compute BMI
bmi = df['weight'] / (df['height'] / 100) ** 2

def weight_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obesity'

df['WeightCategory'] = bmi.apply(weight_category)

def blood_pressure_category(row):
    sys = row['ap_hi']
    dia = row['ap_lo']
    if sys < 120 and dia < 80:
        return 'Normal'
    elif 120 <= sys <= 129 and dia < 80:
        return 'Elevated'
    elif 130 <= sys <= 139 or 80 <= dia <= 89:
        return 'Hypertension Stage 1'
    elif sys >= 140 or dia >= 90:
        return 'Hypertension Stage 2'
    else:
        return 'Uncategorized'

# Apply blood pressure category
# Use axis=1 to apply row-wise

df['BloodPressure'] = df.apply(blood_pressure_category, axis=1)

# Prepare final DataFrame
final_df = df[['id', 'BloodPressure', 'WeightCategory']].rename(columns={'id': 'Id'})

final_df = final_df.sample(n=150, random_state=42)

# Save to new CSV
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"Triadic preprocessing complete. Saved {len(final_df)} entries to {OUTPUT_CSV}.") 