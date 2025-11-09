import random
import matplotlib.pyplot as plt
import pandas as pd
from .config import *
import matplotlib.pyplot as plt
from .utils import get_distance_from_file_name
from optuna.trial import Trial
from optuna.study import Study

def plot_history_from_trial(trial: Trial, metrics=[], study_name=""):
    
    history = trial.user_attrs["history"]
    for k, v in history.items():
        if k == "epoch":
            continue
        if k in metrics:
            plt.plot(history['epoch'], v, label=k)
    plt.title(f"Study: {study_name}, Trial {trial.number}")
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
    plt.gca().xaxis.set_label_text("Epoch")
    plt.legend()
    plt.show()
    
def plot_history_from_study(study: Study, metrics=[]):
    for trial in study.best_trials:
        plot_history_from_trial(trial, metrics, study_name=study.study_name)
    
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
                alpha=0.05,
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

    distances = [*os.listdir(normal_data_dir), *os.listdir(anomalous_data_dir)]
    unique_distances = set([get_distance_from_file_name(x) for x in distances])
    
    normal_data = random.sample(os.listdir(normal_data_dir), k=3)
    anomalous_data = random.sample(os.listdir(anomalous_data_dir), k=3)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    for ax, label in zip(axes[:, 0], ["Normal", "Anomalous"]):
        ax.annotate(f"{label} data",
                    xy=(-0.1, 0.5), xycoords='axes fraction',
                    fontsize=13, ha='right', va='center',
                    rotation=90)
        
    fig.suptitle("Overview of normal and anomalous data", fontsize=16)


    # Plot each sample
    for ax_row, data_files, data_dir in zip(axes, [normal_data, anomalous_data], [normal_data_dir, anomalous_data_dir]):
        for ax, file in zip(ax_row, data_files):
            
            df = pd.read_csv(os.path.join(data_dir, file))
            
            ax.plot(df[["flow", "pressure"]], label=["flow", "pressure"])
            ax.legend(loc="upper right")
            ax.set_title(f"{get_distance_from_file_name(file)} Meters")
            
            xticks = [str(int(tick)) + "s" for tick in list(ax.get_xticks())]

            ax.set_xticks(list(ax.get_xticks()), xticks)
            ax.set_xlim(0, len(df))
            ax.tick_params(left=False, labelleft=False ) 

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig