"""
Urban Pulse - Traffic Analysis Dashboard

Interactive Streamlit dashboard for exploring traffic volume data.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add src to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.visualization import (
    plot_time_series_interactive,
    plot_temperature_vs_traffic_interactive,
    plot_correlation_heatmap_interactive
)

# Page configuration
st.set_page_config(
    page_title="Urban Pulse - Traffic Analysis",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚦 Urban Pulse - Traffic Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load and cache the processed data."""
    try:
        df = pd.read_csv('data/processed/traffic_cleaned.csv', parse_dates=['date_time'])
        return df
    except FileNotFoundError:
        st.error("⚠️ Data file not found! Please run the preprocessing notebook first.")
        return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Date range filter
    if 'date_time' in df.columns:
        min_date = df['date_time'].min().date()
        max_date = df['date_time'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            df = df[(df['date_time'].dt.date >= date_range[0]) & 
                    (df['date_time'].dt.date <= date_range[1])]
    
    # Weather filter
    if 'weather_main' in df.columns:
        weather_options = ['All'] + sorted(df['weather_main'].unique().tolist())
        selected_weather = st.sidebar.selectbox("Filter by Weather", weather_options)
        
        if selected_weather != 'All':
            df = df[df['weather_main'] == selected_weather]
    
    # Rush hour filter
    if 'is_rush_hour' in df.columns:
        rush_filter = st.sidebar.selectbox(
            "Filter by Time Period",
            ['All', 'Rush Hour Only', 'Non-Rush Hour Only']
        )
        
        if rush_filter == 'Rush Hour Only':
            df = df[df['is_rush_hour'] == 1]
        elif rush_filter == 'Non-Rush Hour Only':
            df = df[df['is_rush_hour'] == 0]
    
    # Key Metrics
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_traffic = df['traffic_volume'].mean()
        st.metric("Average Traffic Volume", f"{avg_traffic:,.0f}")
    
    with col2:
        max_traffic = df['traffic_volume'].max()
        st.metric("Peak Traffic Volume", f"{max_traffic:,.0f}")
    
    with col3:
        if 'is_congested' in df.columns:
            congestion_rate = df['is_congested'].mean() * 100
            st.metric("Congestion Rate", f"{congestion_rate:.1f}%")
        else:
            st.metric("Total Records", f"{len(df):,}")
    
    with col4:
        if 'is_weekend' in df.columns:
            weekday_avg = df[df['is_weekend'] == 0]['traffic_volume'].mean()
            weekend_avg = df[df['is_weekend'] == 1]['traffic_volume'].mean()
            diff = ((weekday_avg - weekend_avg) / weekend_avg) * 100
            st.metric("Weekday vs Weekend", f"{diff:.1f}% higher")
        else:
            st.metric("Data Points", f"{len(df):,}")
    
    st.markdown("---")
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs([
        "📅 Time Series Analysis", 
        "🌡️ Temperature vs Traffic",
        "🔗 Feature Correlations"
    ])
    
    with tab1:
        st.header("Traffic Volume Time Series")
        st.write("Interactive time series plot with zoom and pan capabilities. Explore traffic patterns over time.")
        fig = plot_time_series_interactive(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Temporal pattern insights
        st.subheader("📌 Temporal Pattern Insights")
        if 'is_rush_hour' in df.columns and 'hour' in df.columns:
            rush_hours = df[df['is_rush_hour'] == 1]
            non_rush = df[df['is_rush_hour'] == 0]
            rush_avg = rush_hours['traffic_volume'].mean()
            non_rush_avg = non_rush['traffic_volume'].mean()
            rush_pct = ((rush_avg - non_rush_avg) / non_rush_avg) * 100
            
            hourly_avg = df.groupby('hour')['traffic_volume'].mean()
            peak_hour = hourly_avg.idxmax()
            peak_volume = hourly_avg.max()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rush Hour Traffic", f"{rush_avg:,.0f}", f"{rush_pct:.1f}% higher")
            with col2:
                st.metric("Peak Hour", f"{peak_hour}:00", f"{peak_volume:,.0f} vehicles")
            with col3:
                hourly_range = hourly_avg.max() - hourly_avg.min()
                st.metric("Hourly Variation", f"{hourly_range:,.0f}", "vehicles range")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Observations", f"{len(df):,}")
            with col2:
                st.metric("Average Traffic", f"{df['traffic_volume'].mean():,.0f}")
            with col3:
                st.metric("Peak Traffic", f"{df['traffic_volume'].max():,.0f}")
    
    with tab2:
        st.header("Temperature vs Traffic Volume")
        st.write("Interactive scatter plot showing the relationship between temperature and traffic patterns.")
        if 'temp' in df.columns:
            fig = plot_temperature_vs_traffic_interactive(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Temperature and weather insights
            st.subheader("📌 Temperature & Weather Insights")
            temp_celsius = df['temp'] - 273.15
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Temperature", f"{temp_celsius.mean():.1f}°C", 
                         f"Range: {temp_celsius.min():.0f}°C - {temp_celsius.max():.0f}°C")
            
            if 'weather_main' in df.columns:
                adverse_weather = ['Rain', 'Snow', 'Thunderstorm', 'Drizzle', 'Mist', 'Fog']
                clear_weather = ['Clear', 'Clouds']
                
                adverse_avg = df[df['weather_main'].isin(adverse_weather)]['traffic_volume'].mean()
                clear_avg = df[df['weather_main'].isin(clear_weather)]['traffic_volume'].mean()
                
                if len(df[df['weather_main'].isin(adverse_weather)]) > 0:
                    weather_reduction = ((clear_avg - adverse_avg) / clear_avg) * 100
                    with col2:
                        st.metric("Weather Impact", f"{weather_reduction:.1f}%", 
                                 "reduction in adverse weather")
                    with col3:
                        st.metric("Clear Weather Avg", f"{clear_avg:,.0f}", "vehicles")
                else:
                    with col2:
                        st.metric("Avg Temp", f"{temp_celsius.mean():.1f}°C")
                    with col3:
                        st.metric("Max Temp", f"{temp_celsius.max():.1f}°C")
            else:
                with col2:
                    st.metric("Min Temp", f"{temp_celsius.min():.1f}°C")
                with col3:
                    st.metric("Max Temp", f"{temp_celsius.max():.1f}°C")
        else:
            st.warning("Temperature data not available")
    
    with tab3:
        st.header("Feature Correlation Heatmap")
        st.write("Interactive correlation matrix showing relationships between numeric features. Hover for exact values.")
        numeric_cols = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all',
                       'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 1:
            fig = plot_correlation_heatmap_interactive(df, numeric_columns=available_cols)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation insights
            st.subheader("📌 Key Correlations")
            corr_matrix = df[available_cols].corr()
            traffic_corr = corr_matrix['traffic_volume'].sort_values(ascending=False)
            top_corr = traffic_corr[traffic_corr.index != 'traffic_volume'].head(3)
            
            for feature, corr_value in top_corr.items():
                st.write(f"**{feature}**: {corr_value:.3f}")
        else:
            st.warning("Not enough numeric columns for correlation analysis")
    
    # Data Explorer
    st.markdown("---")
    st.header("🔍 Data Explorer")
    
    with st.expander("View Raw Data"):
        num_rows = st.slider("Number of rows to display", 10, 1000, 100)
        st.dataframe(df.head(num_rows), use_container_width=True)
    
    with st.expander("Data Summary"):
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Date Range:**", f"{df['date_time'].min()} to {df['date_time'].max()}")
        st.write("**Columns:**", list(df.columns))
        
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0] if missing.sum() > 0 else pd.Series(["No missing values"]))
    
    # Download button
    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_traffic_data.csv",
        mime="text/csv"
    )

else:
    st.error("""
    ## ⚠️ Data Not Found
    
    Please ensure:
    1. The data preprocessing notebook (`02_data_preprocessing.ipynb`) has been run
    2. The processed data file exists at `data/processed/traffic_cleaned.csv`
    
    Run the preprocessing notebook first to generate the required data file.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>Urban Pulse - Traffic Analysis Dashboard</p>
        <p>Built with Streamlit & Plotly</p>
    </div>
""", unsafe_allow_html=True)

