import random
import matplotlib.pyplot as plt
import pandas as pd
from config import *
import numpy as np
from sklearn.metrics import DetCurveDisplay, det_curve, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from utils import get_best_theshold, get_distance_from_file_name

from utils import get_distance_from_file_name

def visualize_classification(y_true, models_predictions, model_names=None, title="Model Comparison"):
    
    for i, (y_pred, model_name) in enumerate(zip(models_predictions, model_names)):
        y_pred = (y_pred >= get_best_theshold(y_true, y_pred)).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "anomaly"])
        disp.plot()
    
    plt.show()

def visualize_regression(y_true, models_predictions, model_names=None, title="Model Comparison"):
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(models_predictions)))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Scatter plot for each model
    for i, (y_pred, model_name) in enumerate(zip(models_predictions, model_names)):
        axs[0].scatter(
            y_pred,
            y_true,
            color=colors[i],
            label=model_name,
            alpha=0.7,
            edgecolor="k",
        )
        
        axs[1].scatter(
            y_pred,
            y_true - y_pred,
            color=colors[i],
            label=model_name,
            alpha=0.7,
            edgecolor="k",
        )
        
    lims = [
        min(min(y_true), *[min(y_pred) for y_pred in models_predictions]),
        max(max(y_true), *[max(y_pred) for y_pred in models_predictions]),
    ]
    axs[0].set_title("Actual vs. Predicted values")
    axs[0].plot(lims, lims, "--", color="gray", label="Ideal Prediction")
    axs[0].set_ylabel("Actual Values")
    axs[0].set_xlabel("Predicted Values")
    axs[0].legend(loc="best")
    
    axs[1].set_title("Residuals vs. Predicted values")
    axs[1].axhline(0, color="gray", linestyle="--", label="Ideal Prediction")    
    axs[1].legend(loc="best")
    axs[1].set_ylabel("Residuals (actual - predicted)")
    axs[1].set_xlabel("Predicted Values")
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_history_from_study(study):
    history = study.best_trial.user_attrs["history"]
    plt.plot(history['epoch'], history["val_loss"], label="val_loss")
    plt.plot(history['epoch'], history["train_loss"], label="train_loss")
    plt.legend()
    plt.show()
    
def plot_window_predictions(x, y_pred, y_true, window_size, stride, threshold):
    
    leak_detected = False
    plt.figure(figsize=(6, 3), dpi=100)
    for i in range(0, len(y_pred)):
        if y_pred[i] > threshold:
            start = i * stride
            end = i * stride + window_size
            plt.axvspan(
                start,
                end,
                color='green',
                alpha=0.1,
                label='leak detected' if not leak_detected else "_"
            )
            leak_detected = True
    
    plt.plot(x, label=["pressure", "flow"])

    xticks = [str(int(tick)) + "s" for tick in list(plt.xticks()[0])]

    plt.xticks(list(plt.xticks()[0]), xticks)
    plt.xlim(0, len(x))
    plt.tick_params(left=False, labelleft=False ) 

    plt.xlabel("Timestep")
    plt.title(f"LeakNet Predictions - {'No Leak' if y_true == 0 else 'Leak'}")
    plt.legend(loc='upper right')
    plt.show()

def plot_samples(dir: str):
    
    normal_data_dir = os.path.join(dir, "no_leak")
    anomalous_data_dir = os.path.join(dir, "leak")

    unique_distances = [*os.listdir(normal_data_dir), *os.listdir(anomalous_data_dir)]
    unique_distances = set([get_distance_from_file_name(x) for x in unique_distances])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    for ax, label in zip(axes[:, 0], ["Normal", "Anomalous"]):
        ax.annotate(f"{label} data",
                    xy=(-0.1, 0.5), xycoords='axes fraction',
                    fontsize=13, ha='right', va='center',
                    rotation=90)
        
    # Add a main title for the figure
    fig.suptitle("Overview of normal and anomalous data", fontsize=16)

    normal_data = random.sample(os.listdir(normal_data_dir), k=3)
    anomalous_data = random.sample(os.listdir(anomalous_data_dir), k=3)

    for ax_row, data_files, data_dir in zip(axes, [normal_data, anomalous_data], [normal_data_dir, anomalous_data_dir]):
        for ax, file in zip(ax_row, data_files):
            
            df = pd.read_csv(os.path.join(data_dir, file))
            
            ax.plot(df[["flow", "pressure"]], label=["flow", "pressure"])
            ax.legend()
            ax.set_title(f"{get_distance_from_file_name(file)} Meters")
            
            xticks = [str(int(tick)) + "s" for tick in list(ax.get_xticks())]

            ax.set_xticks(list(ax.get_xticks()), xticks)
            ax.set_xlim(0, len(df))
            ax.tick_params(left=False, labelleft=False ) 

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()