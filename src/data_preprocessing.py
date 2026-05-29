import datetime
import pandas as pd

# keep a reusable list of features for training and inference
BASE_FEATURES = ['humidity', 'wind_speed', 'pressure', 'visibility']
DATE_FEATURES = ['month', 'day_of_week', 'day_of_year', 'is_weekend']
ENGINEERED_FEATURES = [
    'humidity_wind_speed',
    'pressure_visibility',
    'humidity_pressure_ratio',
    'wind_speed_visibility_ratio',
    'humidity_sq',
    'wind_speed_sq',
]
FEATURE_COLUMNS = BASE_FEATURES + DATE_FEATURES + ENGINEERED_FEATURES


def _ensure_date_column(df, date_col='date'):
    """Guarantee that a date column exists for date feature engineering."""
    if date_col not in df.columns:
        df[date_col] = pd.Timestamp.now()
    return df


def _add_date_features(df, date_col='date'):
    """Add time-based features from a date column."""
    df = _ensure_date_column(df, date_col=date_col)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    df[date_col] = df[date_col].fillna(pd.Timestamp.now(tz='UTC'))
    df['month'] = df[date_col].dt.month.astype(int)
    df['day_of_week'] = df[date_col].dt.dayofweek.astype(int)
    df['day_of_year'] = df[date_col].dt.dayofyear.astype(int)
    df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    return df


def _add_engineered_features(df):
    """Add interaction and polynomial features to boost model signal."""
    df['humidity_wind_speed'] = df['humidity'] * df['wind_speed']
    df['pressure_visibility'] = df['pressure'] * df['visibility']
    df['humidity_pressure_ratio'] = df['humidity'] / (df['pressure'] + 1e-6)
    df['wind_speed_visibility_ratio'] = df['wind_speed'] / (df['visibility'] + 1e-6)
    df['humidity_sq'] = df['humidity'] ** 2
    df['wind_speed_sq'] = df['wind_speed'] ** 2
    return df


def engineer_features(df, date_col='date'):
    """Apply feature engineering to a DataFrame with date and weather columns."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if date_col not in df.columns:
        df[date_col] = pd.Timestamp.now()

    df = _add_date_features(df, date_col=date_col)
    df = _add_engineered_features(df)
    return df


def preprocess_data(file_path):
    """Load CSV, clean columns, and return features/target for training."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    df.rename(columns={
        'Formatted Date': 'date',
        'Temperature (C)': 'temperature',
        'Humidity': 'humidity',
        'Wind Speed (km/h)': 'wind_speed',
        'Pressure (millibars)': 'pressure',
        'Visibility (km)': 'visibility'
    }, inplace=True)

    df.dropna(inplace=True)
    df = engineer_features(df, date_col='date')

    selected_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not selected_features:
        raise ValueError('No valid features found after engineering. Check dataset schema.')

    X = df[selected_features]
    y = df['temperature']
    return X, y


def prepare_inference_features(humidity, wind_speed, pressure, visibility, date=None):
    """Build a DataFrame with feature-engineered columns for inference."""
    row = {
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure,
        'visibility': visibility,
    }
    if date is not None:
        row['date'] = date

    df = pd.DataFrame([row])
    df = engineer_features(df, date_col='date')

    final_columns = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in final_columns]
    if missing:
        raise ValueError(f"Missing engineered inference features: {missing}")

    return df[final_columns]
