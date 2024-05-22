import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def two_bars_share_x(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    x_name: str,
    y1_name: str,
    y2_name: str,
    title: str,
    x_label: str,
    y1_label: str,
    y2_label: str,
    l1_label: str,
    l2_label: str,
) -> None:
    fig, ax1 = plt.subplots(figsize=(10,5))

    # Create proportion default by date
    ax1.bar(x=df1[x_name], 
            height=df1[y1_name], 
            width=1, color='blue')
    ax1.set_ylabel(y1_label, color='blue')
    ax1.set_xlabel(x_label)

    # Dupe ax
    ax2 = ax1.twinx()

    # Create total by date
    ax2.bar(x=df2[x_name], 
            height=df2[y2_name],
            width=1, color='lime', alpha=0.5)
    ax2.set_ylabel(y2_label, color='lime')

    # Create legend
    custom_lines = [plt.Line2D([0], [0], color='blue', lw=4),
                    plt.Line2D([0], [0], color='lime', lw=4, alpha=0.7)]

    fig.legend(custom_lines, [l1_label, l2_label], loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title(title)
    plt.tight_layout()
    plt.show()