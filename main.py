import argparse
from src.data_preprocessing import preprocess_data
from src.model_training import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the weather forecasting model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/weather.csv",
        help="Path to the weather data CSV file.",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    X, y = preprocess_data(args.data_path)
    print("Training model...")
    train_model(X, y)
    print("Training complete.")