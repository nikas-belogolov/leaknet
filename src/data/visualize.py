import random
import matplotlib.pyplot as plt
import pandas as pd
from config import *

def plot_samples(dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    for ax, label in zip(axes[:, 0], ["Normal", "Anomalous"]):
        ax.annotate(f"{label} data",
                    xy=(-0.2, 0.5), xycoords='axes fraction',
                    fontsize=13, ha='right', va='center',
                    rotation=90)

    # Add a main title for the figure
    fig.suptitle("Overview of normal and anomalous data", fontsize=16)

    normal_data_dir = os.path.join(dir, "no_leak")
    anomalous_data_dir = os.path.join(dir, "leak")

    normal_data = random.sample(os.listdir(normal_data_dir), k=3)
    anomalous_data = random.sample(os.listdir(anomalous_data_dir), k=3)

    for ax, file in zip(axes[0], normal_data):
        df = pd.read_csv(os.path.join(normal_data_dir, file))
        ax.plot(df[["flow", "pressure"]])
        ax.set_title(file)
        
    for ax, file in zip(axes[1], anomalous_data):
        df = pd.read_csv(os.path.join(anomalous_data_dir, file))
        ax.plot(df[["flow", "pressure"]])
        ax.set_title(file)

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()