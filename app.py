import streamlit as st
import pandas as pd
from src.visualization import (
    plot_weather_trends,
    plot_distribution,
    plot_correlation,
    plot_multiple_weather_trends,
    plot_correlation_heatmap,
    plot_box_plot,
    plot_seaborn_boxplot,
    plot_seaborn_scatter,
    plot_seaborn_histogram,
    plot_seaborn_pairplot,
    plot_seaborn_heatmap,
    plot_seaborn_lineplot,
)
from src.prediction import (
    load_model_and_scaler,
    predict_temperature,
    predict_temperature_7_day,
    classify_weather_condition,
)
from src.weather_api import get_current_weather

st.set_page_config(page_title='Weather Forecasting System', layout='wide')

st.markdown("""
<style>
body { background-color: #eaf3ff; }
.stApp { background-color: #eaf3ff; }
</style>
""", unsafe_allow_html=True)

st.title('🌦 Weather Forecasting System')

@st.cache_resource
def cached_load_model_and_scaler():
    try:
        return load_model_and_scaler()
    except FileNotFoundError:
        return None, None

model, scaler = cached_load_model_and_scaler()

if model is None or scaler is None:
    st.error('Model not found. Please train the model first using main.py')
    st.stop()

st.sidebar.header('📍 Location Selection')
location = st.sidebar.text_input('Location', value='New York', placeholder='Enter any city name (e.g., Paris, Tokyo, Sydney)')
prediction_date = st.sidebar.date_input('Prediction date', pd.Timestamp.now().date())

location_adjustment = {
    'New York': 0.3,
    'Los Angeles': 1.2,
    'Chicago': -0.4,
    'Houston': 1.0,
    'Phoenix': 2.3,
    'London': -0.6,
    'Mumbai': 1.5,
    'Tokyo': 0.2,
}
adjustment = location_adjustment.get(location, 0)

st.header('🌤️ Current Weather')
if st.button('Fetch Current Weather'):
    weather_data = get_current_weather(location)
    if weather_data:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Temperature", f"{weather_data['temperature']} °C")
        with col2:
            st.metric("Humidity", f"{weather_data['humidity']:.2f}")
        with col3:
            st.metric("Wind Speed", f"{weather_data['wind_speed']:.1f} km/h")
        with col4:
            st.metric("Pressure", f"{weather_data['pressure']} mb")
        with col5:
            st.metric("Visibility", f"{weather_data['visibility']:.1f} km")
        st.info(f"Weather: {weather_data['description']}")
    else:
        st.error("Failed to fetch current weather. Please check your API key.")

st.markdown('---')

st.header('🔮 Multi-parameter Weather Prediction')
col1, col2, col3, col4 = st.columns(4)
with col1:
    humidity = st.slider('Humidity', 0.0, 1.0, 0.5, step=0.01)
with col2:
    wind_speed = st.slider('Wind Speed (km/h)', 0.0, 60.0, 10.0, step=0.5)
with col3:
    pressure = st.slider('Pressure (millibars)', 900.0, 1100.0, 1013.0, step=0.1)
with col4:
    visibility = st.slider('Visibility (km)', 0.0, 20.0, 10.0, step=0.1)

if st.button('Predict Now'):
    base_temp = predict_temperature(model, scaler, humidity, wind_speed, pressure, visibility, date=pd.Timestamp(prediction_date))
    adjusted_temp = round(base_temp + adjustment, 2)
    condition = classify_weather_condition(adjusted_temp, humidity, wind_speed, visibility)

    st.success(f'🌡 Predicted temperature for {location} on {prediction_date}: {adjusted_temp} °C')
    st.info(f'🧭 Weather condition: {condition}')

st.markdown('---')

st.header('📊 Interactive 7-Day Forecast Dashboard')
with st.expander('Enter 7-day conditions', expanded=True):
    hum_7 = [st.slider(f'Day {i} Humidity', 0.0, 1.0, 0.5, step=0.01, key=f'hum{i}') for i in range(1, 8)]
    wind_7 = [st.slider(f'Day {i} Wind', 0.0, 60.0, 10.0, step=0.5, key=f'wind{i}') for i in range(1, 8)]
    press_7 = [st.slider(f'Day {i} Pressure', 900.0, 1100.0, 1013.0, step=0.1, key=f'press{i}') for i in range(1, 8)]
    vis_7 = [st.slider(f'Day {i} Visibility', 0.0, 20.0, 10.0, step=0.1, key=f'vis{i}') for i in range(1, 8)]

if st.button('Generate 7-Day Forecast'):
    try:
        forecast = predict_temperature_7_day(model, scaler, hum_7, wind_7, press_7, vis_7, start_date=pd.Timestamp(prediction_date))
        df7 = pd.DataFrame({
            'Day': [f'Day {i}' for i in range(1, 8)],
            'Temperature (°C)': [round(t + adjustment, 2) for t in forecast],
            'Humidity': hum_7,
            'Wind speed': wind_7,
            'Pressure': press_7,
            'Visibility': vis_7,
        })
        df7['Condition'] = df7.apply(lambda r: classify_weather_condition(r['Temperature (°C)'], r['Humidity'], r['Wind speed'], r['Visibility']), axis=1)

        st.success('✅ 7-day forecast generated')
        st.dataframe(df7)
        st.line_chart(df7.set_index('Day')['Temperature (°C)'])

        c1, c2, c3 = st.columns(3)
        c1.metric('High', f"{df7['Temperature (°C)'].max():.1f} °C")
        c2.metric('Low', f"{df7['Temperature (°C)'].min():.1f} °C")
        c3.metric('Average', f"{df7['Temperature (°C)'].mean():.1f} °C")
    except Exception as e:
        st.error(f'Error generating 7-day forecast: {e}')

st.markdown('---')
st.header('📈 Historical Weather Trends')

uploaded_file = st.file_uploader('Upload weather CSV', type=['csv'])
source = 'data/weather.csv'
if uploaded_file is not None:
    source = uploaded_file

try:
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    if 'Formatted Date' in df.columns:
        df = df.rename(columns={'Formatted Date': 'date'})
    if 'date' not in df.columns:
        st.warning('CSV requires date/Formatted Date column.')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        # Normalize timezone-aware timestamps to UTC and remove tz info for plotting consistency
        df['date'] = df['date'].dt.tz_convert('UTC').dt.tz_localize(None)
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not numeric_columns:
            st.warning('No numeric columns to plot.')
        else:
            st.subheader('Single Variable Analysis')
            col1, col2 = st.columns(2)
            with col1:
                selected = st.selectbox('Select feature to plot', numeric_columns, index=0)
                plot_type = st.selectbox('Plot Type', ['Line Plot', 'Histogram', 'Box Plot'], key='single')
            with col2:
                if plot_type == 'Line Plot':
                    fig = plot_weather_trends(df.copy(), column_to_plot=selected)
                elif plot_type == 'Histogram':
                    fig = plot_seaborn_histogram(df, column_to_plot=selected)
                elif plot_type == 'Box Plot':
                    fig = plot_seaborn_boxplot(df, column_to_plot=selected)
                st.pyplot(fig)
            
            st.subheader('Multi-Variable Analysis')
            col3, col4 = st.columns(2)
            with col3:
                multi_plot_type = st.selectbox('Multi-Variable Plot Type', 
                                             ['Correlation Heatmap', 'Pair Plot', 'Multiple Trends'], 
                                             key='multi')
            with col4:
                if multi_plot_type == 'Correlation Heatmap':
                    fig = plot_seaborn_heatmap(df)
                elif multi_plot_type == 'Pair Plot':
                    selected_cols = st.multiselect('Select columns for pair plot', numeric_columns, 
                                                 default=numeric_columns[:4] if len(numeric_columns) >= 4 else numeric_columns)
                    if selected_cols:
                        fig = plot_seaborn_pairplot(df, columns=selected_cols)
                    else:
                        st.warning('Select at least one column.')
                        fig = None
                elif multi_plot_type == 'Multiple Trends':
                    selected_trends = st.multiselect('Select columns for trends', numeric_columns, 
                                                   default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns)
                    if selected_trends:
                        fig = plot_multiple_weather_trends(df.copy(), columns_to_plot=selected_trends)
                    else:
                        st.warning('Select at least one column.')
                        fig = None
                if fig:
                    st.pyplot(fig)
            
            st.subheader('Relationship Analysis')
            col5, col6 = st.columns(2)
            with col5:
                x_var = st.selectbox('X Variable', numeric_columns, index=0, key='x')
                y_var = st.selectbox('Y Variable', numeric_columns, index=1 if len(numeric_columns) > 1 else 0, key='y')
                rel_plot_type = st.selectbox('Relationship Plot Type', ['Scatter Plot', 'Line Plot'], key='rel')
            with col6:
                hue_var = st.selectbox('Hue (optional)', ['None'] + categorical_columns, key='hue')
                hue = None if hue_var == 'None' else hue_var
                if rel_plot_type == 'Scatter Plot':
                    fig = plot_seaborn_scatter(df, x_col=x_var, y_col=y_var, hue=hue)
                elif rel_plot_type == 'Line Plot':
                    fig = plot_seaborn_lineplot(df, x_col='date', y_col=y_var, hue=hue)
                st.pyplot(fig)

except FileNotFoundError:
    st.warning('No historical file found. Upload CSV or add data/weather.csv.')
except Exception as e:
    st.error(f'Error loading historical data: {e}')