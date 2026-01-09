"""
Visualization Module for Urban Pulse Project

This module contains functions for creating various types of visualizations
for exploratory data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# Optional Plotly imports for interactive visualizations
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_traffic_distribution(df: pd.DataFrame, 
                             volume_column: str = 'traffic_volume',
                             save_path: Optional[str] = None) -> None:
    """
    Create histogram showing traffic volume distribution.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column (default: 'traffic_volume')
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(df[volume_column], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Traffic Volume', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Traffic Volume Distribution (Histogram)', fontsize=14, fontweight='bold')
    axes[0].axvline(df[volume_column].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df[volume_column].mean():.0f}')
    axes[0].axvline(df[volume_column].median(), color='green', linestyle='--', 
                    label=f'Median: {df[volume_column].median():.0f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # KDE Plot
    axes[1].hist(df[volume_column], bins=50, density=True, alpha=0.5, color='steelblue', label='Histogram')
    df[volume_column].plot(kind='kde', ax=axes[1], color='darkblue', linewidth=2, label='KDE')
    axes[1].set_xlabel('Traffic Volume', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Traffic Volume Distribution (KDE)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_traffic_by_weekday(df: pd.DataFrame,
                            volume_column: str = 'traffic_volume',
                            day_column: str = 'day_of_week',
                            save_path: Optional[str] = None) -> None:
    """
    Create boxplot showing traffic volume by day of week.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column
    day_column : str, optional
        Name of day of week column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df_box = df.copy()
    if df_box[day_column].dtype != 'object':
        df_box['day_name'] = df_box[day_column].map(dict(enumerate(day_names)))
    else:
        df_box['day_name'] = df_box[day_column]
    
    sns.boxplot(data=df_box, x='day_name', y=volume_column, ax=axes[0], palette='Set2')
    axes[0].set_xlabel('Day of Week', fontsize=12)
    axes[0].set_ylabel('Traffic Volume', fontsize=12)
    axes[0].set_title('Traffic Volume by Day of Week (Boxplot)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Bar chart with means
    mean_by_day = df_box.groupby('day_name')[volume_column].mean()
    axes[1].bar(mean_by_day.index, mean_by_day.values, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Day of Week', fontsize=12)
    axes[1].set_ylabel('Average Traffic Volume', fontsize=12)
    axes[1].set_title('Average Traffic Volume by Day of Week', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_time_series(df: pd.DataFrame,
                    date_column: str = 'date_time',
                    volume_column: str = 'traffic_volume',
                    save_path: Optional[str] = None) -> None:
    """
    Create time series plot showing traffic volume over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str, optional
        Name of datetime column
    volume_column : str, optional
        Name of traffic volume column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    df_ts = df.copy()
    df_ts = df_ts.sort_values(date_column)
    
    # Full time series
    axes[0].plot(df_ts[date_column], df_ts[volume_column], linewidth=0.5, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Traffic Volume', fontsize=12)
    axes[0].set_title('Traffic Volume Over Time (Full Series)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Rolling average
    if len(df_ts) > 24:  # Only if we have enough data points
        window = min(24, len(df_ts) // 10)  # 24-hour rolling average
        df_ts['rolling_mean'] = df_ts[volume_column].rolling(window=window).mean()
        axes[1].plot(df_ts[date_column], df_ts[volume_column], linewidth=0.3, alpha=0.5, 
                    color='lightblue', label='Raw Data')
        axes[1].plot(df_ts[date_column], df_ts['rolling_mean'], linewidth=2, 
                    color='darkblue', label=f'{window}-Hour Rolling Average')
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Traffic Volume', fontsize=12)
        axes[1].set_title(f'Traffic Volume with {window}-Hour Rolling Average', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame,
                            numeric_columns: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> None:
    """
    Create correlation heatmap for numeric features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    numeric_columns : list, optional
        List of numeric columns to include. If None, auto-selects.
    save_path : str, optional
        Path to save the figure
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target if it's in the list (we'll add it back separately)
        if 'is_congested' in numeric_columns:
            numeric_columns.remove('is_congested')
    
    # Calculate correlation
    corr_data = df[numeric_columns + ['is_congested'] if 'is_congested' in df.columns else numeric_columns].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap: Traffic Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_temperature_vs_traffic(df: pd.DataFrame,
                               temp_column: str = 'temp',
                               volume_column: str = 'traffic_volume',
                               save_path: Optional[str] = None) -> None:
    """
    Create scatter plot showing relationship between temperature and traffic.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    temp_column : str, optional
        Name of temperature column (assumed to be in Kelvin)
    volume_column : str, optional
        Name of traffic volume column
    save_path : str, optional
        Path to save the figure
    """
    # Filter out invalid temperature values (0 or extremely low)
    # Temperature in Kelvin should be > 200K (approximately -73°C)
    df_clean = df.copy()
    df_clean = df_clean[(df_clean[temp_column] > 200) & (df_clean[temp_column] < 320)]
    
    if len(df_clean) == 0:
        print("Warning: No valid temperature data after filtering")
        return
    
    # Convert temperature from Kelvin to Celsius for better interpretation
    df_clean['temp_celsius'] = df_clean[temp_column] - 273.15
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot with Celsius
    scatter = axes[0].scatter(df_clean['temp_celsius'], df_clean[volume_column], alpha=0.5, 
                            c=df_clean[volume_column], cmap='viridis', s=20)
    axes[0].set_xlabel('Temperature (°C)', fontsize=12)
    axes[0].set_ylabel('Traffic Volume', fontsize=12)
    axes[0].set_title('Temperature vs Traffic Volume (Scatter Plot)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0], label='Traffic Volume')
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line (only for valid data range)
    z = np.polyfit(df_clean['temp_celsius'], df_clean[volume_column], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_clean['temp_celsius'].min(), df_clean['temp_celsius'].max(), 100)
    y_trend = p(x_trend)
    
    # Clip trend line to non-negative traffic volumes
    y_trend = np.maximum(y_trend, 0)
    
    axes[0].plot(x_trend, y_trend, "r--", alpha=0.8, linewidth=2, 
                label=f'Trend: y={z[0]:.2f}x+{z[1]:.0f}')
    axes[0].legend()
    axes[0].set_ylim(bottom=0)  # Ensure y-axis starts at 0
    
    # Boxplot by temperature bins (using Celsius with quantile-based bins)
    temp_min = df_clean['temp_celsius'].min()
    temp_max = df_clean['temp_celsius'].max()
    
    # Use quantiles for better binning
    bins = [
        df_clean['temp_celsius'].quantile(0),
        df_clean['temp_celsius'].quantile(0.2),
        df_clean['temp_celsius'].quantile(0.4),
        df_clean['temp_celsius'].quantile(0.6),
        df_clean['temp_celsius'].quantile(0.8),
        df_clean['temp_celsius'].quantile(1.0)
    ]
    
    labels = ['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot']
    df_clean['temp_bin'] = pd.cut(df_clean['temp_celsius'], bins=bins, labels=labels, include_lowest=True)
    
    sns.boxplot(data=df_clean, x='temp_bin', y=volume_column, ax=axes[1], palette='coolwarm')
    axes[1].set_xlabel('Temperature Range', fontsize=12)
    axes[1].set_ylabel('Traffic Volume', fontsize=12)
    axes[1].set_title('Traffic Volume by Temperature Range', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(bottom=0)  # Ensure y-axis starts at 0
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_congestion_by_hour(df: pd.DataFrame,
                           hour_column: str = 'hour',
                           save_path: Optional[str] = None) -> None:
    """
    Create bar chart showing congestion levels by hour of day.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    hour_column : str, optional
        Name of hour column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average traffic by hour
    hourly_avg = df.groupby(hour_column)['traffic_volume'].mean()
    axes[0].bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Hour of Day', fontsize=12)
    axes[0].set_ylabel('Average Traffic Volume', fontsize=12)
    axes[0].set_title('Average Traffic Volume by Hour of Day', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(0, 24))
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Congestion rate by hour (if is_congested exists)
    if 'is_congested' in df.columns:
        congestion_rate = df.groupby(hour_column)['is_congested'].mean() * 100
        axes[1].bar(congestion_rate.index, congestion_rate.values, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Hour of Day', fontsize=12)
        axes[1].set_ylabel('Congestion Rate (%)', fontsize=12)
        axes[1].set_title('Congestion Rate by Hour of Day', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(0, 24))
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_rush_hour_comparison(df: pd.DataFrame,
                             volume_column: str = 'traffic_volume',
                             save_path: Optional[str] = None) -> None:
    """
    Create violin plot comparing rush hour vs non-rush hour traffic.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Violin plot
    df_plot = df.copy()
    df_plot['Rush Hour'] = df_plot['is_rush_hour'].map({0: 'Non-Rush Hour', 1: 'Rush Hour'})
    
    sns.violinplot(data=df_plot, x='Rush Hour', y=volume_column, ax=axes[0], palette='Set2')
    axes[0].set_xlabel('Time Period', fontsize=12)
    axes[0].set_ylabel('Traffic Volume', fontsize=12)
    axes[0].set_title('Traffic Volume: Rush Hour vs Non-Rush Hour (Violin Plot)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Swarm plot (sample if too many points)
    df_sample = df_plot.sample(min(1000, len(df_plot))) if len(df_plot) > 1000 else df_plot
    sns.swarmplot(data=df_sample, x='Rush Hour', y=volume_column, ax=axes[1], 
                 size=2, alpha=0.5, palette='Set2')
    axes[1].set_xlabel('Time Period', fontsize=12)
    axes[1].set_ylabel('Traffic Volume', fontsize=12)
    axes[1].set_title('Traffic Volume: Rush Hour vs Non-Rush Hour (Swarm Plot)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_weather_impact(df: pd.DataFrame,
                       weather_column: str = 'weather_main',
                       volume_column: str = 'traffic_volume',
                       save_path: Optional[str] = None) -> None:
    """
    Create bar chart showing traffic volume by weather condition.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    weather_column : str, optional
        Name of weather column
    volume_column : str, optional
        Name of traffic volume column
    save_path : str, optional
        Path to save the figure
    """
    if weather_column not in df.columns:
        print(f"Warning: Column '{weather_column}' not found in DataFrame")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average traffic by weather
    weather_avg = df.groupby(weather_column)[volume_column].mean().sort_values(ascending=False)
    axes[0].barh(weather_avg.index, weather_avg.values, color='teal', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Average Traffic Volume', fontsize=12)
    axes[0].set_ylabel('Weather Condition', fontsize=12)
    axes[0].set_title('Average Traffic Volume by Weather Condition', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Count by weather
    weather_count = df[weather_column].value_counts()
    axes[1].barh(weather_count.index, weather_count.values, color='skyblue', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Number of Observations', fontsize=12)
    axes[1].set_ylabel('Weather Condition', fontsize=12)
    axes[1].set_title('Number of Observations by Weather Condition', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def create_summary_statistics_plot(df: pd.DataFrame,
                                  volume_column: str = 'traffic_volume',
                                  save_path: Optional[str] = None) -> None:
    """
    Create a summary statistics visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column
    save_path : str, optional
        Path to save the figure
    """
    stats = df[volume_column].describe()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Statistics bar chart
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    stat_values = [stats['count'], stats['mean'], stats['std'], stats['min'], 
                  stats['25%'], stats['50%'], stats['75%'], stats['max']]
    
    axes[0].bar(stat_names, stat_values, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Summary Statistics for Traffic Volume', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Boxplot with statistics annotations
    bp = axes[1].boxplot(df[volume_column], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[1].set_ylabel('Traffic Volume', fontsize=12)
    axes[1].set_title('Traffic Volume Distribution (Boxplot)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    axes[1].text(1.1, stats['min'], f"Min: {stats['min']:.0f}", fontsize=9, verticalalignment='bottom')
    axes[1].text(1.1, stats['25%'], f"Q1: {stats['25%']:.0f}", fontsize=9, verticalalignment='center')
    axes[1].text(1.1, stats['50%'], f"Median: {stats['50%']:.0f}", fontsize=9, verticalalignment='center')
    axes[1].text(1.1, stats['75%'], f"Q3: {stats['75%']:.0f}", fontsize=9, verticalalignment='center')
    axes[1].text(1.1, stats['max'], f"Max: {stats['max']:.0f}", fontsize=9, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


# ============================================================================
# Interactive Visualizations (Plotly)
# ============================================================================

def plot_time_series_interactive(df: pd.DataFrame,
                                 date_column: str = 'date_time',
                                 volume_column: str = 'traffic_volume') -> go.Figure:
    """
    Create interactive time series plot with zoom and pan capabilities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str, optional
        Name of datetime column
    volume_column : str, optional
        Name of traffic volume column
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualizations. Install with: pip install plotly")
    
    df_ts = df.copy()
    df_ts = df_ts.sort_values(date_column)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Traffic Volume Over Time', 'Traffic with Rolling Average'),
        vertical_spacing=0.1
    )
    
    # Full time series
    fig.add_trace(
        go.Scatter(
            x=df_ts[date_column],
            y=df_ts[volume_column],
            mode='lines',
            name='Traffic Volume',
            line=dict(color='steelblue', width=1),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Rolling average
    if len(df_ts) > 24:
        window = min(24, len(df_ts) // 10)
        df_ts['rolling_mean'] = df_ts[volume_column].rolling(window=window).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df_ts[date_column],
                y=df_ts[volume_column],
                mode='lines',
                name='Raw Data',
                line=dict(color='lightblue', width=1),
                opacity=0.3
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_ts[date_column],
                y=df_ts['rolling_mean'],
                mode='lines',
                name=f'{window}-Hour Rolling Average',
                line=dict(color='darkblue', width=2)
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Traffic Volume", row=1, col=1)
    fig.update_yaxes(title_text="Traffic Volume", row=2, col=1)
    
    fig.update_layout(
        title_text="Traffic Volume Time Series Analysis",
        height=700,
        hovermode='x unified'
    )
    
    return fig


def plot_correlation_heatmap_interactive(df: pd.DataFrame,
                                         numeric_columns: Optional[List[str]] = None) -> go.Figure:
    """
    Create interactive correlation heatmap with hover details.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    numeric_columns : list, optional
        List of numeric column names to include
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualizations. Install with: pip install plotly")
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation
    corr_matrix = df[numeric_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation"),
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title_text="Feature Correlation Heatmap",
        height=600,
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig


def plot_temperature_vs_traffic_interactive(df: pd.DataFrame,
                                            temp_column: str = 'temp',
                                            volume_column: str = 'traffic_volume') -> go.Figure:
    """
    Create interactive scatter plot showing temperature vs traffic with zoom capabilities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    temp_column : str, optional
        Name of temperature column (assumed to be in Kelvin)
    volume_column : str, optional
        Name of traffic volume column
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualizations. Install with: pip install plotly")
    
    # Filter out invalid temperature values
    df_clean = df.copy()
    df_clean = df_clean[(df_clean[temp_column] > 200) & (df_clean[temp_column] < 320)]
    
    if len(df_clean) == 0:
        raise ValueError("No valid temperature data after filtering")
    
    # Convert temperature from Kelvin to Celsius
    df_clean['temp_celsius'] = df_clean[temp_column] - 273.15
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Temperature vs Traffic Volume', 'Traffic by Temperature Range'),
        specs=[[{"type": "scatter"}, {"type": "box"}]]
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=df_clean['temp_celsius'],
            y=df_clean[volume_column],
            mode='markers',
            name='Traffic',
            marker=dict(
                color=df_clean[volume_column],
                colorscale='Viridis',
                size=5,
                opacity=0.6,
                showscale=True,
                colorbar=dict(title="Traffic Volume", x=1.15)
            ),
            hovertemplate='<b>Temperature:</b> %{x:.1f}°C<br>' +
                         '<b>Traffic:</b> %{y:.0f} vehicles<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add trend line
    z = np.polyfit(df_clean['temp_celsius'], df_clean[volume_column], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_clean['temp_celsius'].min(), df_clean['temp_celsius'].max(), 100)
    y_trend = p(x_trend)
    y_trend = np.maximum(y_trend, 0)
    
    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name=f'Trend: y={z[0]:.2f}x+{z[1]:.0f}',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Trend Line</b><extra></extra>'
        ),
        row=1, col=1
    )
    
    # Boxplot by temperature bins
    bins = [
        df_clean['temp_celsius'].quantile(0),
        df_clean['temp_celsius'].quantile(0.2),
        df_clean['temp_celsius'].quantile(0.4),
        df_clean['temp_celsius'].quantile(0.6),
        df_clean['temp_celsius'].quantile(0.8),
        df_clean['temp_celsius'].quantile(1.0)
    ]
    
    labels = ['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot']
    df_clean['temp_bin'] = pd.cut(df_clean['temp_celsius'], bins=bins, labels=labels, include_lowest=True)
    
    for temp_bin in labels:
        bin_data = df_clean[df_clean['temp_bin'] == temp_bin][volume_column]
        if len(bin_data) > 0:
            fig.add_trace(
                go.Box(
                    y=bin_data,
                    name=temp_bin,
                    boxmean='sd',
                    hovertemplate=f'<b>{temp_bin}</b><br>' +
                                 '<b>Traffic:</b> %{y:.0f} vehicles<br>' +
                                 '<extra></extra>'
                ),
                row=1, col=2
            )
    
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature Range", row=1, col=2)
    fig.update_yaxes(title_text="Traffic Volume", row=1, col=1)
    fig.update_yaxes(title_text="Traffic Volume", row=1, col=2)
    
    fig.update_layout(
        title_text="Temperature Impact on Traffic",
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig
