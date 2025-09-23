import matplotlib.pyplot as plt

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
