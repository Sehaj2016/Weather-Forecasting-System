import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv('WEATHER_API_KEY')
if not API_KEY:
    raise RuntimeError('WEATHER_API_KEY is not set in environment. Add it to .env or environment variables.')

BASE_URL = "https://api.weatherapi.com/v1/current.json"

def get_current_weather(city):
    """
    Fetch current weather data for a given city using WeatherAPI.

    Parameters:
    city (str): Name of the city

    Returns:
    dict: Weather data including temperature, humidity, wind speed, pressure, visibility
    """
    params = {
        'key': API_KEY,
        'q': city,
        'aqi': 'no'  # No air quality data
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract relevant data from WeatherAPI response
        current = data['current']
        weather_data = {
            'temperature': current['temp_c'],
            'humidity': current['humidity'] / 100,  # Convert to 0-1 scale
            'wind_speed': current['wind_kph'],
            'pressure': current['pressure_mb'],
            'visibility': current['vis_km'],
            'description': current['condition']['text']
        }

        return weather_data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    except KeyError as e:
        print(f"Error parsing weather data: {e}")
        return None