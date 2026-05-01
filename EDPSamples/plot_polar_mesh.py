import numpy as np
import matplotlib.pyplot as plt

def plot_polar_mesh(vertices: np.ndarray, triangles: np.ndarray, pole: str = "north", ax=None):
    """
    Plots a triangular mesh in a polar projection centered on the north or south pole.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 2)
        Array of (longitude, latitude) in degrees.
    triangles : np.ndarray, shape (M, 3)
        Each row contains indices of the triangle corners.
    pole : str, "north" or "south"
        Which pole to center the projection on.
    ax : matplotlib axes, optional
        Existing axes to plot on.
    Returns
    -------
    ax : matplotlib axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
    else:
        fig = ax.figure

    # Convert to radians for polar plotting
    lons = np.deg2rad(vertices[:, 0])
    lats = vertices[:, 1]

    # For north pole: r = 90 - lat, for south pole: r = 90 + lat
    if pole.lower() == "north":
        r = 90 - lats
        theta = lons
        ax.set_theta_zero_location('N')
        ax.set_ylim(0, 90-np.min(vertices[:,1]))
        ax.set_title("Polar Mesh (North Pole)", va='bottom')
    elif pole.lower() == "south":
        r = 90 + lats
        theta = lons
        ax.set_theta_zero_location('S')
        ax.set_ylim(0, 90-np.max(vertices[:,1]))
        ax.set_title("Polar Mesh (South Pole)", va='bottom')
    else:
        raise ValueError("pole must be 'north' or 'south'")

    # Plot each triangle
    for tri in triangles:
        pts = np.array([theta[tri], r[tri]])
        pts = np.column_stack((theta[tri], r[tri]))
        # Close the triangle
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], color='k', lw=0.7, alpha=0.7)
        ax.fill(pts[:, 0], pts[:, 1], facecolor='C0', alpha=0.25, edgecolor=None)

    # Plot the vertices
    ax.scatter(theta, r, s=15, c='red', zorder=5, label='Vertices')

    # Optional: grid and labels for degrees
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_yticks(np.arange(0, 91, 15))
    ax.grid(True, alpha=0.4)
    ax.legend(loc='lower right')
    if pole.lower() == "north":
        ax.set_yticklabels([f"{90-x}°" for x in ax.get_yticks()])
        ax.set_ylim(0, 90-np.min(vertices[:,1]))
    elif pole.lower() == "south":
        ax.set_yticklabels([f"{-90+x}°" for x in ax.get_yticks()])
        ax.set_ylim(0, 90+np.max(vertices[:,1]))
    else:
        raise ValueError("pole must be 'north' or 'south'")

    return ax