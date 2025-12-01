import pandas as pd

def create_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Extract date features
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofweek'] = df['date'].dt.dayofweek

    # Lag features
    df['aqi_lag1'] = df['aqi'].shift(1)
    df['aqi_lag2'] = df['aqi'].shift(2)

    # Rolling mean
    df['aqi_roll3'] = df['aqi'].rolling(window=3).mean()

    # Encode city (label encoding)
    df['city'] = df['city'].astype('category').cat.codes

    # Drop any rows with NaN after shifting
    df = df.dropna()

    # Save
    df.to_csv(output_path, index=False)
    print("Feature engineered data saved to:", output_path)


if __name__ == "__main__":
    create_features("../data/cleaned.csv", "../data/features.csv")
