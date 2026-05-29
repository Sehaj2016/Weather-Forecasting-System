from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    print("Model Performance")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("R2 Score:", r2_score(y_test, predictions))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/temperature_model.pkl")

    print("Model saved successfully!")