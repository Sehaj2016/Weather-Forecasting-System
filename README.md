# Weather Forecasting System

A machine learning-based weather forecasting application built with Streamlit.

## Features

- Temperature prediction based on humidity, wind speed, pressure, and visibility
- 7-day forecast dashboard
- Weather condition classification
- Current weather data integration via WeatherAPI (supports any location worldwide)

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate the virtual environment: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## API Integration

To fetch current weather data, you need a WeatherAPI key:

1. Sign up at [WeatherAPI](https://www.weatherapi.com/)
2. Get your free API key
3. Add the key to `.env` in the project root:

   ```
   WEATHER_API_KEY=your_api_key_here
   ```

4. Install dependencies: `pip install -r requirements.txt`

`src/weather_api.py` now loads `WEATHER_API_KEY` using `python-dotenv`.

## Usage

1. Train the model: `python main.py`
2. Run the app: `streamlit run app.py`
3. Enter any city name in the location field to get weather data and predictions for that location

## Project Structure

- `app.py`: Streamlit web application
- `main.py`: Model training script
- `src/`: Source code modules
  - `data_preprocessing.py`: Data cleaning and preprocessing
  - `model_training.py`: Model training and saving
  - `prediction.py`: Prediction functions
  - `visualization.py`: Plotting functions
  - `weather_api.py`: Weather API integration
- `data/`: Weather data CSV
- `models/`: Trained model files
