import pandas as pd

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

    # Convert types
    df = df.infer_objects(copy=False)

    # Remove negative values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: None if x < 0 else x)

    # Interpolate numeric columns only (no warning)
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

    # Fill remaining NaN
    df = df.ffill().bfill()

    df = df.drop_duplicates()

    df.to_csv(output_path, index=False)
    print("Cleaned dataset saved to:", output_path)


if __name__ == "__main__":
    clean_data("../data/aqi_raw.csv", "../data/cleaned.csv")
