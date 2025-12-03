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
            
            # Set y-axis label for leftmost column
            if j == 0:
                ax.set_ylabel(parameters[i].replace(" [", "\n["), fontsize=10)
            
            ax.set_xlabel(parameters[j].replace(" [", "\n["), fontsize=10)
            ax.tick_params(axis="both", labelsize=9)

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


def _detect_gaps(tunnel_length, threshold=1.0):
    """
    Detect gaps in tunnel length data.
    
    Parameters:
    -----------
    tunnel_length : array-like
        Tunnel length values
    threshold : float
        Minimum gap size to detect (in meters)
    
    Returns:
    --------
    list of tuples
        Each tuple contains (start_position, end_position) of a gap
    """
    gaps = []
    for i in range(len(tunnel_length) - 1):
        diff = tunnel_length.iloc[i + 1] - tunnel_length.iloc[i]
        if diff > threshold:
            gaps.append((tunnel_length.iloc[i], tunnel_length.iloc[i + 1]))
    return gaps


def _break_at_gaps(x, y, gaps):
    """
    Insert NaN values at gap locations to break line plots.
    
    Parameters:
    -----------
    x : array-like
        X-axis values (tunnel length)
    y : array-like
        Y-axis values (parameter values)
    gaps : list of tuples
        Gap intervals from _detect_gaps
    
    Returns:
    --------
    tuple of (x_broken, y_broken)
        Arrays with NaN inserted at gap positions
    """
    if len(gaps) == 0:
        return np.array(x), np.array(y)
    
    # Create a copy as lists for manipulation
    x_list = list(x.values if hasattr(x, 'values') else x)
    y_list = list(y.values if hasattr(y, 'values') else y)
    
    # Track insertions (process from end to beginning to maintain indices)
    for gap_start, gap_end in reversed(gaps):
        # Find the index right after gap_start
        insert_idx = None
        for i in range(len(x_list)):
            if x_list[i] > gap_start:
                insert_idx = i
                break
        
        # Insert NaN to break the line at the gap
        if insert_idx is not None and insert_idx > 0:
            x_list.insert(insert_idx, np.nan)
            y_list.insert(insert_idx, np.nan)
    
    return np.array(x_list), np.array(y_list)


def _add_collapse_shading(ax, dataframe):
    """
    Add red shading for collapse regions across the full height of the plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to add shading to
    dataframe : pd.DataFrame
        Data containing tunnel length and collapse labels
    """
    x = dataframe["Tunnellength [m]"]
    collapse = dataframe["collapse"]
    
    # Fill regions where collapse == 1 across full plot height
    ax.fill_between(x, 0, 1, where=(collapse == 1), 
                     color='red', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)


def _plot_param(ax, dataframe, parameter, gaps, show_gap_markers=False):
    """
    Plot a parameter with gap handling and collapse shading.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    dataframe : pd.DataFrame
        Data containing tunnel length and parameter
    parameter : str
        Parameter name to plot
    gaps : list of tuples
        Gap intervals
    show_gap_markers : bool
        Whether to show X markers over gaps
    """
    x = dataframe["Tunnellength [m]"]
    y = dataframe[parameter]
    
    # Add collapse shading first (so it's in background)
    _add_collapse_shading(ax, dataframe)
    
    # Break lines at gaps
    x_broken, y_broken = _break_at_gaps(x, y, gaps)
    
    ax.plot(x_broken, y_broken, color="black", linewidth=0.5)
    ax.set_ylabel(parameter)
    ax.set_xlim(left=x.min(), right=x.max())
    ax.grid(alpha=0.5)
    
    # Add gap markers if requested
    if show_gap_markers:
        for gap_start, gap_end in gaps:
            ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3, hatch='///', zorder=10)


def plot_tbm_parameters(df, gap_threshold=1.0, show_gap_markers=False):
    """
    Plot TBM parameters along tunnel length with gap handling and collapse visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data containing TBM parameters
    gap_threshold : float
        Minimum gap size in meters to detect and handle (default: 1.0)
    show_gap_markers : bool
        Whether to show hatched regions over data gaps (default: False)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Detect gaps in the data
    gaps = _detect_gaps(df["Tunnellength [m]"], threshold=gap_threshold)
    
    fig, axs = plt.subplots(ncols=1, nrows=5, figsize=(18, 8), sharex=True)

    _plot_param(axs[0], df, "penetration\n[mm/rev]", gaps, show_gap_markers)
    _plot_param(axs[1], df, "advance rate\n[mm/min]", gaps, show_gap_markers)
    _plot_param(axs[2], df, "cutterhead rotations\n[rpm]", gaps, show_gap_markers)
    _plot_param(axs[3], df, "thrust\n[kN]", gaps, show_gap_markers)
    _plot_param(axs[4], df, "cutterhead torque\n[kNm]", gaps, show_gap_markers)
    axs[4].set_xlabel("Tunnellength [m]")
    
    # Add legend for collapse regions on the top plot
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.2, label='Collapse')]
    axs[0].legend(handles=legend_elements, loc='upper right')

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
    cell_colors: dict[str, str] | None = None,
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
        Colormap name. Default is 'Greys'. Ignored if cell_colors is provided.
    dpi : int, optional
        DPI for saved figures. Default is 300
    cell_colors : dict[str, str], optional
        Dictionary with keys 'tn_color', 'fp_color', 'fn_color', 'tp_color'
        to specify custom colors for each cell. If provided, overrides cmap.
        Maps to: TR (True Regular), FC (False Collapse), FR (False Regular), TC (True Collapse)
        Example: {'tn_color': '#90EE90', 'fp_color': '#FFD700', 'fn_color': '#FF6B6B', 'tp_color': '#4ECDC4'}

    Returns:
    --------
    plt.Figure
        The matplotlib figure object

    Example:
    --------
    >>> class_mapping = {0: "regular\nexcavation", 1: "collapse"}
    >>> fig = plot_tbm_confusion_matrix(y_test, y_pred, class_mapping, "RandomForest")
    """

    # Calculate confusion matrix for display values (normalized to 'all' for percentages)
    cm_display_values = confusion_matrix(y_true, y_pred, normalize='all')
    cm_display = np.round(cm_display_values * 100, 1)  # Round to 1 decimal place for percentages
    
    # Calculate confusion matrix for color opacity (normalized by true labels)
    cm_for_colors = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Use custom colors if provided, otherwise use colormap
    if cell_colors is not None:
        # Create a custom colored confusion matrix with opacity based on values
        # For binary classification: [[TN, FP], [FN, TP]]
        color_matrix = np.array([
            [cell_colors.get('tn_color', '#90EE90'), cell_colors.get('fp_color', '#FFD700')],
            [cell_colors.get('fn_color', '#FF6B6B'), cell_colors.get('tp_color', '#4ECDC4')]
        ])
        
        # Normalize values to determine opacity (0-1 range)
        # Use 'true' normalization for colors (each row sums to 1)
        opacity_values = cm_for_colors
        
        # Display each cell with its specific color and opacity
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                # Convert hex color to RGB and apply opacity
                import matplotlib.colors as mcolors
                base_color = mcolors.to_rgb(color_matrix[i, j])
                opacity = opacity_values[i, j]
                
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                          facecolor=base_color, 
                                          alpha=opacity,
                                          edgecolor='white', linewidth=2))
        
        # Set axis limits
        ax.set_xlim(-0.5, cm_display.shape[1] - 0.5)
        ax.set_ylim(cm_display.shape[0] - 0.5, -0.5)
    else:
        # Display confusion matrix as image with colormap
        # Use 'true' normalization for colors
        cm_for_colormap = confusion_matrix(y_true, y_pred, normalize='true')
        im = ax.imshow(cm_for_colormap, interpolation="nearest", cmap=getattr(plt.cm, cmap))
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
        # title=f'Confusion Matrix{f" ({model_name})" if model_name else ""}',
    )

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add percentage annotations in cells
    if show_percentages:
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                value = cm_display[i, j]
                
                # Choose text color based on cell opacity/value (using 'true' normalization for consistency)
                # For custom colors, base on normalized value; for colormap, use original logic
                if cell_colors is not None:
                    # Use 'true' normalization for text color decision
                    opacity = cm_for_colors[i, j]
                    text_color = "white" if opacity > 0.5 else "black"
                else:
                    # Use 'true' normalization for colormap
                    opacity = cm_for_colors[i, j]
                    text_color = "white" if opacity > 0.5 else "black"

                # Format text - always show percentages normalized to 'all'
                text = f"({value:.1f}%)"

                ax.text(j, i, text, ha="center", va="center", color=text_color)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    return fig
