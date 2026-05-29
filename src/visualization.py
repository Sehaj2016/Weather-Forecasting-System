import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_weather_trends(df, column_to_plot='temperature'):
    """
    Plots trends for a selected column from a DataFrame.
    """
    # Convert date column to datetime objects
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.sort_values('date')
    else:
        raise ValueError("Dataset must contain a 'date' column.")

    if column_to_plot not in df.columns:
        raise ValueError(f"Column '{column_to_plot}' not found in the dataset.")

    fig, ax = plt.subplots(figsize=(10, 5))
    # Plotting selected column vs Date
    ax.plot(df['date'], df[column_to_plot], label=column_to_plot, color='tab:red', linewidth=1)
    ax.set_title(f"Historical Trend for {column_to_plot}")
    ax.set_xlabel("Date")
    ax.set_ylabel(column_to_plot)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

def plot_distribution(df, column_to_plot):
    """
    Plots the distribution (histogram) of a selected column.
    """
    if column_to_plot not in df.columns:
        raise ValueError(f"Column '{column_to_plot}' not found in the dataset.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[column_to_plot], bins=30, color='tab:blue', edgecolor='black', alpha=0.7)
    ax.set_title(f"Distribution of {column_to_plot}")
    ax.set_xlabel(column_to_plot)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_correlation(df, col_x, col_y):
    """
    Plots a scatter plot to show correlation between two columns.
    """
    if col_x not in df.columns or col_y not in df.columns:
        raise ValueError(f"Columns '{col_x}' or '{col_y}' not found in the dataset.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[col_x], df[col_y], alpha=0.5, color='tab:green')
    ax.set_title(f"Correlation between {col_x} and {col_y}")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    plt.tight_layout()
    return fig

def plot_multiple_weather_trends(df, columns_to_plot):
    """
    Plots trends for multiple selected columns from a DataFrame on the same chart.
    """
    if not isinstance(columns_to_plot, list):
        raise TypeError("columns_to_plot must be a list of column names.")

    # Convert date column to datetime objects
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.sort_values('date')
    else:
        raise ValueError("Dataset must contain a 'date' column.")

    fig, ax = plt.subplots(figsize=(12, 6))

    for column in columns_to_plot:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")
        ax.plot(df['date'], df[column], label=column, linewidth=1)

    ax.set_title("Weather Trends for Multiple Variables")
    ax.set_xlabel("Date")
    ax.set_ylabel("Values")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for numeric columns.
    """
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation.")

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap='coolwarm', interpolation='nearest')
    fig.colorbar(cax)

    ticks = range(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap")

    # Loop over data dimensions and create text annotations.
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black")

    plt.tight_layout()
    return fig

def plot_box_plot(df, column_to_plot):
    """
    Plots a box plot for a selected column.
    """
    if column_to_plot not in df.columns:
        raise ValueError(f"Column '{column_to_plot}' not found in the dataset.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(df[column_to_plot].dropna())
    ax.set_title(f"Box Plot of {column_to_plot}")
    ax.set_ylabel(column_to_plot)
    plt.tight_layout()
    return fig

def plot_seaborn_boxplot(df, column_to_plot, hue=None):
    """
    Plots a seaborn box plot for a selected column, optionally with hue.
    """
    if column_to_plot not in df.columns:
        raise ValueError(f"Column '{column_to_plot}' not found in the dataset.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=column_to_plot, ax=ax, hue=hue)
    ax.set_title(f"Seaborn Box Plot of {column_to_plot}")
    plt.tight_layout()
    return fig

def plot_seaborn_scatter(df, x_col, y_col, hue=None):
    """
    Plots a seaborn scatter plot between two columns, optionally with hue.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' or '{y_col}' not found in the dataset.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
    ax.set_title(f"Seaborn Scatter Plot: {x_col} vs {y_col}")
    plt.tight_layout()
    return fig

def plot_seaborn_histogram(df, column_to_plot, hue=None):
    """
    Plots a seaborn histogram for a selected column, optionally with hue.
    """
    if column_to_plot not in df.columns:
        raise ValueError(f"Column '{column_to_plot}' not found in the dataset.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=column_to_plot, hue=hue, ax=ax, kde=True)
    ax.set_title(f"Seaborn Histogram of {column_to_plot}")
    plt.tight_layout()
    return fig

def plot_seaborn_pairplot(df, columns=None, hue=None):
    """
    Plots a seaborn pairplot for selected columns.
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            raise ValueError("No numeric columns found for pairplot.")
        columns = numeric_df.columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
    
    pair_df = df[columns].copy()
    if hue and hue in df.columns:
        pair_df[hue] = df[hue]
    
    fig = sns.pairplot(pair_df, hue=hue).fig
    fig.suptitle("Seaborn Pairplot", y=1.02)
    plt.tight_layout()
    return fig

def plot_seaborn_heatmap(df):
    """
    Plots a seaborn correlation heatmap.
    """
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for heatmap.")
    
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Seaborn Correlation Heatmap")
    plt.tight_layout()
    return fig

def plot_seaborn_lineplot(df, x_col, y_col, hue=None):
    """
    Plots a seaborn line plot.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' or '{y_col}' not found in the dataset.")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
    ax.set_title(f"Seaborn Line Plot: {y_col} over {x_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig