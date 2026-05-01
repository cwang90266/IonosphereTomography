import numpy as np
import matplotlib.pyplot as plt

def plot_tri_mesh(vertices: np.ndarray, triangles: np.ndarray, ax=None):
    """
    Plot triangular mesh patches and their vertices on a map.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 2)
        Each row is (longitude, latitude) of a vertex.
    triangles : np.ndarray, shape (M, 3)
        Each row contains indices of the vertices composing one triangle.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Plot each triangle
    for tri in triangles:
        tri_pts = vertices[tri]
        # close the polygon by adding the first point to the end
        polygon = np.vstack([tri_pts, tri_pts[0]])
        ax.plot(polygon[:,0], polygon[:,1], 'k-', lw=0.8, alpha=0.7)
        ax.fill(polygon[:,0], polygon[:,1], facecolor='C0', alpha=0.25, edgecolor=None)

    # Plot the vertices
    ax.scatter(vertices[:,0], vertices[:,1], color='red', s=12, label='Vertices', zorder=5)

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Triangular Mesh")
    ax.autoscale()
    ax.legend()
    ax.grid(True, alpha=0.4)
    return ax