import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn.preprocessing import StandardScaler
import umap
from hdbscan import HDBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE

def investigate_umap(
    table_path,
    data_keys,
    label_column,
    labels_name,
    labels_color_map,
    n_neighbors_list=[15, 15, 15],
    min_dist_list=[0.1, 0.3, 1],
    cluster_method=None,  # Accepts 'GMM', 'HDBSCAN', or None
    n_components_gmm=5,
    min_cluster_size_hdbscan=30,
    min_samples_hdbscan=1,
    axis_label_fontsize=18,
    tick_fontsize=18,
    title_fontsize=19,
    legend_fontsize=15
):
    """
    Visualizes UMAP dimensionality reduction results for a high-dimensional dataset and 
    optionally applies clustering (GMM or HDBSCAN) in the reduced space.

    Parameters
    ----------
    table_path : str
        Path to the FITS file containing the dataset.
    data_keys : list of str
        List of column names to use as input features for UMAP.
    label_column : str
        Column name containing original cluster assignments for coloring true label plots.
    labels_name : dict
        Dictionary mapping numerical cluster IDs to string labels (e.g. {1: 'GS/E', 2: 'Splash'}).
    labels_color_map : dict
        Dictionary mapping string labels to matplotlib-compatible color codes.
    n_neighbors_list : list of int, optional
        List of UMAP `n_neighbors` values, one per column of the plot grid.
    min_dist_list : list of float, optional
        List of UMAP `min_dist` values, one per column of the plot grid.
    cluster_method : str or None, optional
        If specified, applies unsupervised clustering in UMAP space. Options:
        - 'GMM': Gaussian Mixture Model clustering (requires `n_components_gmm`).
        - 'HDBSCAN': HDBSCAN clustering (requires `min_cluster_size_hdbscan` and `min_samples_hdbscan`).
        - None: disables clustering, shows only UMAP colored by original labels.
    n_components_gmm : int, optional
        Number of clusters to fit for GMM if `cluster_method='GMM'`. Default is 5.
    min_cluster_size_hdbscan : int, optional
        Minimum cluster size for HDBSCAN. Only used if `cluster_method='HDBSCAN'`.
    min_samples_hdbscan : int, optional
        Minimum samples for HDBSCAN. Only used if `cluster_method='HDBSCAN'`.
    axis_label_fontsize : int, optional
        Font size for axis labels.
    tick_fontsize : int, optional
        Font size for axis tick labels.
    title_fontsize : int, optional
        Font size for row titles.
    legend_fontsize : int, optional
        Font size for legend and text annotations.

    Returns
    -------
    None
        Displays matplotlib figures with UMAP projections and clustering overlays if enabled.
    """
    # Extrct the atropy table from the fits path
    tbl = Table.read(table_path, format='fits')

    # Extract the values from the table depending on the data keys
    X = tbl[data_keys].to_pandas().values

    # Apply standard scaling each of the columns dimensions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Extract the labels assigned during high dimensional GMM - XD Clustering to help visualize the clusters
    labels = tbl[label_column]
    named_labels = np.array([labels_name[int(i)] for i in labels])
    unique_names = sorted(set(named_labels))

    # Assign the number of graph rows based wether a clustering method is used or not
    n_cols = len(n_neighbors_list)
    n_rows = 2 if cluster_method in ['GMM', 'HDBSCAN'] else 1

    # Plottig the UMAP results
    # Create a grid of subplots
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.075)
    axes = np.empty((n_rows, n_cols), dtype=object)

    for idx, (n_n, m_d) in enumerate(zip(n_neighbors_list, min_dist_list)):
        # For all UMAP configurations, apply the UMAP data reduction, obtaining the 2D coordinates
        reducer = umap.UMAP(n_components=2, n_neighbors=n_n, min_dist=m_d, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)

        # Plot the UMAP results colour coded by the GMM labels from the XD clustering - no analysis has been done on the UMAP results at this stage
        ax_top = fig.add_subplot(gs[0, idx])
        axes[0, idx] = ax_top
        for name in unique_names:
            mask = named_labels == name
            ax_top.scatter(X_umap[mask, 0], X_umap[mask, 1],
                           label=name, color=labels_color_map[name], s=10, alpha=0.7)
        ax_top.set_xlabel("UMAP-1", fontsize=axis_label_fontsize)
        ax_top.set_ylabel("UMAP-2", fontsize=axis_label_fontsize)
        ax_top.tick_params(axis='both', labelsize=tick_fontsize)

        # Add UMAP params to top-right of each plot
        ax_top.text(0.99, 0.98, f"n_neighbors={n_n}\nmin_dist={m_d}",
                    transform=ax_top.transAxes, ha='right', va='top',
                    fontsize=legend_fontsize,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # If a clustering method is selected, we apply clustering to the UMAP coordinates and compare the results visually to see the plausibility of the clusters results
        # GMM Clustering
        if cluster_method == 'GMM':
            gmm = GaussianMixture(n_components=n_components_gmm, random_state=9)
            cluster_labels = gmm.fit_predict(X_umap)
        
        # HDBSCAN Clustering
        elif cluster_method == 'HDBSCAN':
            hdb = HDBSCAN(min_cluster_size=min_cluster_size_hdbscan, min_samples=min_samples_hdbscan)
            cluster_labels = hdb.fit_predict(X_umap)
        else:
            cluster_labels = None

        # Plot the clustering results - the color/ labeling of these is `random` each time so no labels are used or assigned - expecially with APOGEE where we obtain poor results
        if cluster_labels is not None:
            ax_bot = fig.add_subplot(gs[1, idx])
            axes[1, idx] = ax_bot
            ax_bot.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels, cmap='tab10', s=10, alpha=0.7)
            ax_bot.set_xlabel("UMAP-1", fontsize=axis_label_fontsize)
            ax_bot.set_ylabel("UMAP-2", fontsize=axis_label_fontsize)
            ax_bot.tick_params(axis='both', labelsize=tick_fontsize)

    # Add titles to the plots
    if cluster_method:
        fig.text(0.5, 0.52, f"{cluster_method} Clustering Labels", ha='center', va='top', fontsize=title_fontsize)
    fig.text(0.5, 1.0, "High Dimensional XD Labels", ha='center', va='bottom', fontsize=title_fontsize)

    # Add XD labels to the top-left the UMAP plot
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels_, title='True Label', loc='upper left', fontsize=legend_fontsize)

    # if cluster_method == 'HDBSCAN' add clustering hyperparameters to the top-left of each plot
    if cluster_method == 'HDBSCAN':
        for ax in axes[1, :]:
            ax.text(0.01, 0.98,
                    f"min_cluster_size={min_cluster_size_hdbscan}\nmin_samples={min_samples_hdbscan}",
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=legend_fontsize,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    plt.show()



def investigate_tsne(
    table_path,
    data_keys,
    perplexities,
    learning_rates,
    label_column='max_gauss',
    labels_name=None,
    labels_color_map=None,
    axis_label_fontsize=14,
    tick_fontsize=12,
    legend_fontsize=10,
    title_fontsize=14
):
    """
    Visualizes t-SNE dimensionality reduction results across multiple configurations.

    Parameters
    ----------
    table_path : str
        Path to the FITS file containing the data table.
    data_keys : list of str
        Column names to use as input features for dimensionality reduction.
    perplexities : list of int
        List of perplexity values for each t-SNE configuration.
    learning_rates : list of float
        List of learning rate values for each t-SNE configuration.
    label_column : str, optional
        Column name representing true GMM cluster labels. Default is 'max_gauss'.
    labels_name : dict, optional
        Mapping from numeric GMM component indices to descriptive cluster names.
    labels_color_map : dict, optional
        Mapping from descriptive cluster names to color codes.
    axis_label_fontsize : int, optional
        Font size for axis labels.
    tick_fontsize : int, optional
        Font size for axis ticks.
    legend_fontsize : int, optional
        Font size for the legend.
    title_fontsize : int, optional
        Font size for plot titles.

    Returns
    -------
    None
    """
    assert len(perplexities) == len(learning_rates), "perplexities and learning_rates must be same length"

    # Load the data from the FITS file 
    tbl = Table.read(table_path, format='fits')

    # Extract the values from the table depending on the data keys
    X = tbl[data_keys].to_pandas().values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Extract the labels assigned during high dimensional GMM - XD Clustering to help visualize the clusters
    labels = tbl[label_column]

    # cross reference the labels with the names and colors provided
    named_labels = np.array([labels_name[int(i)] for i in labels])
    unique_names = sorted(set(named_labels))

    # Plotting the t-SNE results
    fig, axes = plt.subplots(1, len(perplexities), figsize=(6 * len(perplexities), 5))
    if len(perplexities) == 1:
        axes = [axes]

    # Create a grid of subplots
    for idx, (perp, lr) in enumerate(zip(perplexities, learning_rates)):
        # For all t-SNE configurations, apply the t-SNE data reduction, obtaining the 2D coordinates
        tsne = TSNE(n_components=2, perplexity=perp, learning_rate=lr, random_state=42, init='pca', n_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)

        # Plot the t-SNE results colour coded by the GMM labels from the XD clustering - no analysis is done on the t-SNE results
        ax = axes[idx]
        for name in unique_names:
            mask = named_labels == name
            ax.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                label=name,
                color=labels_color_map[name],
                s=10
            )

        # Set the titles and labels
        ax.set_xlabel("t-SNE-1", fontsize=axis_label_fontsize)
        ax.set_ylabel("t-SNE-2", fontsize=axis_label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Add configuration parameters to the top-left of each plot
        config_text = f"Perplexity={perp}\nLR={lr}"
        ax.text(0.02, 0.98, config_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=tick_fontsize,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))

    # Shared legend in top-right of the last plot
    handles, labels_ = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels_, title='GMM Component', loc='upper right', fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()