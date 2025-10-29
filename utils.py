from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="whitegrid", palette="pastel")


# --------- Feature Assessment and Visualization ---------
def feature_stats(feature_name, df):
    '''
    compute statistics for a given feature in the dataframe.
    args:
        feature_name (str)
        df (pd.DataFrame)
    returns:
        dict: dictionary containing various statistics about the feature.
    '''

    feature = df[feature_name]
    stats = {}
    # As we know there are no categorical features, we only compute numeric stats
    feature_type = 'numeric' if is_numeric_dtype(feature) else 'categorical'
    stats['type'] = feature_type
    stats['total_count'] = len(feature)
    distinct_count = feature.nunique(dropna=True)
    stats['distinct_count(%)'] = (distinct_count / len(feature)) * 100   
    missing_count = feature.isnull().sum()
    stats['missing_count(%)'] = (missing_count / len(feature)) * 100
    memory_size = feature.memory_usage(deep=True)
    stats['memory_size(bytes)'] = memory_size

    # Compute descriptive statistics
    stats['mean'] = feature.mean()
    stats['std'] = feature.std()
    if feature.mean() != 0:
        stats['coefficient of variation'] = stats['std'] / stats['mean']
    stats['min'] = feature.min()
    stats['25%'] = feature.quantile(0.25)
    stats['median(50%)'] = feature.median()
    stats['75%'] = feature.quantile(0.75)
    stats['max'] = feature.max()
    if feature.mean() != 0:
        stats['coefficient of variation'] = stats['std'] / stats['mean']
    
    return stats


def explore_feature(feature_name, df, bins=30, bar_width=3):
    '''
    
    '''
    stats = feature_stats(feature_name, df)
    feature = df[feature_name].dropna()
    stats_df = pd.DataFrame(stats.items(), columns=["Metric", "Value"])
    stats_df["Value"] = stats_df["Value"].apply(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x)

    fig, (ax_table, ax_plot) = plt.subplots(1, 2, figsize=(12, 4))

    # histogram plot
    sns.histplot(feature, bins=bins, kde=True, ax=ax_plot, shrink=bar_width, color="#9C7EAD")
    ax_plot.set_title(f"{feature_name} Distribution")
    ax_plot.set_xlabel(feature_name)
    ax_plot.set_ylabel("Count")

    # stats table
    ax_table.axis("off")
    ax_table.text(0.02, 1.05, "Summary statistics", fontsize=11, weight="bold", family="DejaVu Sans")
    y_start = 0.95
    y_step = 1.0 / (len(stats_df) + 2)
    font_family = "DejaVu Sans"
    ax_table.text(0.02, y_start, "Metric", weight="bold", fontsize=10, family=font_family)
    ax_table.text(0.98, y_start, "Value", weight="bold", fontsize=10, family=font_family, ha="right")

    y = y_start - y_step
    for i, (metric, value) in enumerate(stats_df.values):
        y_mid = y - (y_step / 2)
        ax_table.text(0.02, y_mid, metric, fontsize=9.5, family=font_family, color="#333", va="center")
        ax_table.text(0.98, y_mid, value, fontsize=9.5, family=font_family, color="#333", ha="right", va="center")
        ax_table.plot([0.02, 0.98], [y - y_step, y - y_step], color="#D3D3D3", lw=0.6)
        y -= y_step

    plt.tight_layout()
    plt.show()


def explore_features_grid(df, feature_list, bins=30, bar_width=1.2, n_cols=3):
    """
    Runs explore_feature() for each column and arranges all outputs
    in a single grid (3 per row) — without modifying explore_feature().
    """
    total = len(feature_list)
    n_rows = int(np.ceil(total / n_cols))
    images = []

    # --- Temporarily suppress plt.show() ---
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None

    for feature in feature_list:
        explore_feature(feature, df, bins=bins, bar_width=bar_width)
        fig_temp = plt.gcf()

        buf = BytesIO()
        fig_temp.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img = plt.imread(buf)
        buf.close()
        images.append(img)

        plt.close(fig_temp)

    # Restore normal plt.show()
    plt.show = original_show

    # --- Display all captured plots together ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5))
    axes = np.atleast_2d(axes).flatten()

    for i, img in enumerate(images):
        ax = axes[i]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(feature_list[i], fontsize=11, weight="bold", pad=10)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    plt.show()

