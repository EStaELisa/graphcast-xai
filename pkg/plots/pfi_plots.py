import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance_bar(summary_df: pd.DataFrame, title: str = "Permutation Feature Importance"):
    """
    Plot permutation feature importance as a bar chart with error bars.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with columns: 'feature', 'delta_rmse', 'sem_rmse'.
    title : str, optional
        Title for the plot.
    """
    # Sort by importance
    plot_df = summary_df.sort_values("delta_rmse", ascending=False)

    # Optional pretty feature names
    rename_map = {
        "geopotential": "Geopotential",
        "specific_humidity": "Specific Humidity",
        "temperature": "Temperature",
        "u_component_of_wind_v_component_of_wind": "Wind Speed",
        "vertical_velocity": "Vertical Velocity",
        "2m_temperature": "2m Temperature",
        "10m_u_component_of_wind_10m_v_component_of_wind": "10m Wind Speed",
        "total_precipitation_6hr": "Total Precipitation (6h)",
        "mean_sea_level_pressure": "Mean Sea Level Pressure",
    }
    plot_df["pretty_feature"] = plot_df["feature"].replace(rename_map)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(
        plot_df["pretty_feature"],
        plot_df["delta_rmse"],
        yerr=plot_df["sem_rmse"],
        capsize=5
    )

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Δ RMSE (↑ = more important)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
