
import re
import numpy as np

def lower_slash_format(s):
    return s.lower().replace(" / ", "/") 

def normalize_rgb(color):
    return color/255
    
def set_equal_plot_axes(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Reshape a voxel grid to a cube
def reshape_voxel_grid(sparse, dims, place_top=False, dtype=np.bool, cur_shape = None):
    if sparse.ndim!=2 or sparse.shape[0]!=3:
        raise ValueError('voxel_data is wrong shape; should be 3xN array.')
    if np.isscalar(dims):
        dims = [dims]*3
    dims = np.atleast_2d(dims).T
    # truncate to integers
    xyz = sparse.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dims), 0)
    xyz = xyz[:,valid_ix]
    # If shape specified, center model
    if cur_shape is not None:
        x_dim = dims[0][0]
        x_offset = (x_dim - cur_shape[0]) // 2
        xyz[0] = xyz[0] + x_offset
        z_dim = dims[2][0]
        z_offset = (z_dim - cur_shape[2]) // 2
        xyz[2] = xyz[2] + z_offset
    if place_top:
        # Take desired y dimension
        y_dim = dims[1][0]
        y_diff = (y_dim - 1) - np.max(xyz[1]) 
        xyz[1] = xyz[1] + y_diff
    out = np.zeros(dims.flatten(), dtype=dtype)
    out[tuple(xyz)] = True
    return out