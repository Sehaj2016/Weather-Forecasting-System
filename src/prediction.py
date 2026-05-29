import joblib
import pandas as pd
from src.data_preprocessing import prepare_inference_features

def load_model_and_scaler():
    """Loads the model and scaler from disk."""
    model = joblib.load("models/temperature_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


def classify_weather_condition(temperature_c, humidity=None, wind_speed=None, visibility=None):
    if temperature_c <= 5:
        base = 'Cold'
    elif temperature_c <= 15:
        base = 'Cool'
    elif temperature_c <= 25:
        base = 'Warm'
    else:
        base = 'Hot'
    modifiers = []
    if humidity is not None and humidity > 0.8:
        modifiers.append('Humid')
    if wind_speed is not None and wind_speed > 25:
        modifiers.append('Windy')
    if visibility is not None and visibility < 3:
        modifiers.append('Low visibility')
    if modifiers:
        return base + ' (' + ', '.join(modifiers) + ')'
    return base


def predict_temperature(model, scaler, humidity, wind_speed, pressure, visibility, date=None):
    """Predicts temperature for one row input by applying engineered features and scaling."""
    data = prepare_inference_features(humidity, wind_speed, pressure, visibility, date=date)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return round(float(prediction[0]), 2)


def predict_temperature_7_day(model, scaler, humidity, wind_speed, pressure, visibility, start_date=None):
    """Predict temperatures for the next 7 days by applying feature engineering and scaling."""
    if start_date is None:
        start_date = pd.Timestamp.now().normalize()

    inputs = {
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure,
        'visibility': visibility,
    }

    df = pd.DataFrame(
        {
            k: v if isinstance(v, (list, tuple, pd.Series)) else [v] * 7
            for k, v in inputs.items()
        }
    )

    if len(df) != 7:
        raise ValueError("Each input list must have exactly 7 values for a 7-day forecast.")

    df_features = []
    for i in range(7):
        this_date = start_date + pd.Timedelta(days=i)
        row = df.iloc[i]
        row_features = prepare_inference_features(
            float(row['humidity']),
            float(row['wind_speed']),
            float(row['pressure']),
            float(row['visibility']),
            date=this_date,
        )
        df_features.append(row_features)

    df_features = pd.concat(df_features, ignore_index=True)
    df_scaled = scaler.transform(df_features)
    predictions = model.predict(df_scaled)
    return [round(float(pred), 2) for pred in predictions]
