from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def pairplot(
    df,
    parameters: list[str],
    target: str | None = "collapse",
    bins: int = 30,
    axis_fontsize: int = 8,
    **kwargs,
):
    """
    Create a pair plot to visualize relationships between variables in a dataset.

    Parameters:
    data (DataFrame): The input data for plotting.
    **kwargs: Additional keyword arguments for customization.

    Returns:
    None
    """
    n_params = len(parameters)

    fig, axs = plt.subplots(nrows=n_params, ncols=n_params, figsize=(18, 18))

    df_collaps = df[df[target] == 1]
    df_regular = df[df[target] != 1]

    for i in range(n_params):
        for j in range(n_params):
            ax: plt.Axes = axs[i, j]
            if i == j:
                _, xbins, _ = ax.hist(
                    df[parameters[i]],
                    color="grey",
                    bins=30,
                    edgecolor="black",
                    alpha=0.5,
                    density=True,
                )
                ax.hist(
                    df_collaps[parameters[i]],
                    bins=xbins,
                    color="red",
                    alpha=0.5,
                    edgecolor="black",
                    zorder=2,
                    density=True,
                )
            else:
                ax.scatter(
                    df_regular[parameters[j]],
                    df_regular[parameters[i]],
                    color="grey",
                    edgecolor="black",
                    alpha=0.3,
                    s=1,
                )
                ax.scatter(
                    df_collaps[parameters[j]],
                    df_collaps[parameters[i]],
                    color="red",
                    alpha=1,
                    s=2,
                )
                ax.set_ylabel(parameters[i].replace(" [", "\n["), fontsize=8)
            ax.set_xlabel(parameters[j].replace(" [", "\n["), fontsize=8)
            ax.tick_params(axis="both", labelsize=8)

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


def _plot_param(ax, dataframe, parameter):
    ax.plot(dataframe["Tunnellength [m]"], dataframe[parameter], color="black")
    ax.set_ylabel(parameter)
    ax.set_xlim(
        left=dataframe["Tunnellength [m]"].min(),
        right=dataframe["Tunnellength [m]"].max(),
    )
    # ax.legend()
    ax.grid(alpha=0.5)


def plot_tbm_parameters(df):
    """ """
    fig, axs = plt.subplots(ncols=1, nrows=8, figsize=(18, 10), sharex=True)

    _plot_param(axs[0], df, "penetration\n[mm/rev]")
    _plot_param(axs[1], df, "advance rate\n[mm/min]")
    _plot_param(axs[2], df, "cutterhead rotations\n[rpm]")
    _plot_param(axs[3], df, "thrust\n[kN]")
    _plot_param(axs[4], df, "cutterhead torque\n[kNm]")
    _plot_param(axs[5], df, "Field Penetration Index")
    _plot_param(axs[6], df, "drilling efficiency index\nTPI")
    _plot_param(axs[7], df, "collapse")
    axs[7].set_xlabel("Tunnellength [m]")

    return fig


def plot_tbm_confusion_matrix(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    class_mapping: dict[int, str],
    model_name: str = "",
    figsize: tuple = (5, 5),
    normalize: str = "true",
    show_percentages: bool = True,
    cmap: str = "Greys",
    dpi: int = 300,
) -> plt.Figure:
    """
    Create a confusion matrix plot matching the style from A_main.py and preliminary_tests.py.

    This function creates a confusion matrix with percentage annotations and custom styling
    that matches the original implementation used in the TBM collapse prediction analysis.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_mapping : dict[int, str]
        Mapping of class numbers to class names.
        Example: {0: "regular", 1: "collapse"}
    model_name : str, optional
        Name of the model for the title. Default is ""
    figsize : tuple, optional
        Figure size. Default is (5, 5)
    normalize : str, optional
        Normalization option for confusion matrix ('true', 'pred', 'all', or None).
        Default is 'true' (normalize by true labels)
    show_percentages : bool, optional
        Whether to show percentage annotations in cells. Default is True
    cmap : str, optional
        Colormap name. Default is 'Greys'
    dpi : int, optional
        DPI for saved figures. Default is 300

    Returns:
    --------
    plt.Figure
        The matplotlib figure object

    Example:
    --------
    >>> class_mapping = {0: "regular\nexcavation", 1: "collapse"}
    >>> fig = plot_tbm_confusion_matrix(y_test, y_pred, class_mapping, "RandomForest")
    """

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Convert to percentages if normalized
    if normalize is not None:
        cm_display = np.round(cm * 100, 1)  # Round to 1 decimal place for percentages
    else:
        cm_display = cm.astype(int)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Display confusion matrix as image
    im = ax.imshow(cm_display, interpolation="nearest", cmap=getattr(plt.cm, cmap))

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Get class labels from mapping
    class_labels = [class_mapping.get(i, str(i)) for i in sorted(class_mapping.keys())]

    # Set labels and title
    ax.set(
        xticks=np.arange(len(class_labels)),
        yticks=np.arange(len(class_labels)),
        xticklabels=class_labels,
        yticklabels=class_labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=f'Confusion Matrix{f" ({model_name})" if model_name else ""}',
    )

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add percentage annotations in cells
    if show_percentages:
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                value = cm_display[i, j]

                # Choose text color based on cell value (white for dark cells, black for light cells)
                text_color = "white" if value > (cm_display.max() / 2) else "black"

                # Format text based on whether values are percentages or counts
                if normalize is not None:
                    text = f"({value:.1f}%)"
                else:
                    text = f"{int(value)}"

                ax.text(j, i, text, ha="center", va="center", color=text_color)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    return fig
