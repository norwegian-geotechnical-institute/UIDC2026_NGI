from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def pairplot(data,
             parameters: list[str],
             target: str | None = None,
             bins: int = 30,
             axis_fontsize: int = 8,
             **kwargs):
    """
    Create a pair plot to visualize relationships between variables in a dataset.

    Parameters:
    data (DataFrame): The input data for plotting.
    **kwargs: Additional keyword arguments for customization.

    Returns:
    None
    """
    df = data.copy()

    df_params = df[parameters]
    df_target = df[target] if target else None

    if target:
        df_collaps = df[df[target] == 1]
        df_regular = df[df[target] != 1]

    n_params = len(parameters)

    fig = plt.figure(figsize=kwargs["figsize"] if "figsize" in kwargs else (10, 10))

    n_figure = 1

    for i in range(n_params):
        for j in range(n_params):
            ax = fig.add_subplot(n_params, n_params, n_figure)
            if i == j:
                n, xbins, _ = ax.hist(df_params.iloc[:, i], color='grey', bins=bins, edgecolor='black')

                if target:
                    n_target, _, _ = ax.hist(df_collaps.iloc[:, i], bins=xbins, color='red', alpha=0.7, edgecolor='black', zorder=2)

            else:
                ax.scatter(df_params.iloc[:, i], df_params.iloc[:, j], c=kwargs.get('c', 'grey'), alpha=0.5, s=2)
                ax.set_ylabel(parameters[j].replace(' [', '\n['),
                            fontsize=axis_fontsize)
            ax.set_xlabel(parameters[i].replace(' [', '\n['),
                            fontsize=axis_fontsize)
            ax.tick_params(axis='both', labelsize=axis_fontsize)
            n_figure += 1

    return fig


def add_black_grid_lines(ax: plt.Axes, conf_matrix: np.ndarray) -> None:
    """
    Adds black grid lines around each square in the confusion matrix.

    Parameters:
    ax (plt.Axes): The axes object of the plot.
    conf_matrix (np.ndarray): The confusion matrix data.
    """
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor("black")
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(np.arange(conf_matrix.shape[1]) + 0.5, minor=True)
    ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=True)
    ax.tick_params(which="minor", size=0)


def plot_confusion_matrix(
    y_true: list[Any],
    y_pred: list[Any],
    class_mapping: dict[int, str],
    normalize: str = "true",
    add_black_lines: bool = True,
) -> plt.Figure:
    """
    Plots a confusion matrix with options for customization.

    Parameters:
    y_true (list[Any]): True labels.
    y_pred (list[Any]): Predicted labels.
    class_mapping (dict[int, str]): Mapping of class numbers to class names.
    normalize (str, optional): Normalization option for confusion matrix. Defaults to 'true'.
    add_black_lines (bool, optional): Whether to add black grid lines around each square. Defaults to True.

    Example class_mapping:

    soil_classification = {
    1: "gravel",
    4: "sand to gravel",
    3: "coarse grained organic soils",
    5: "sand",
    2: "fine grained organic soils",
    6: "silt to fine sand",
    7: "clay to silt"
    }

    """
    # Update the labels using the mapping table with map function
    class_labels = list(class_mapping.values())
    class_label_numbers = list(class_mapping.keys())

    # Generate the confusion matrix with labels in the desired order
    conf_matrix = confusion_matrix(
        y_true, y_pred, labels=class_label_numbers, normalize=normalize
    )

    # Round the values in the confusion matrix to a maximum of 3 digits behind the comma
    conf_matrix = np.round(conf_matrix, 3)

    # Visualise the confusion matrix with an optional thin black line around each square
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=class_labels
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    # Toggle black grid lines on or off
    if add_black_lines:
        add_black_grid_lines(ax, conf_matrix)

    # Ensure labels align correctly with ticks
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_yticklabels(class_labels)

    for text in disp.text_.ravel():
        text.set_fontsize(12)  # Adjust the size to your liking

    plt.title("Confusion Matrix (recall for each class on the diagonal)")
    plt.tight_layout()
    return fig