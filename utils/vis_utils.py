import PIL.Image as pil
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
def convert_array_to_pil(depth_map):
    # Normalize depth map
    normalized_depth_map = depth_map / np.max(depth_map)

    cmap = plt.get_cmap('hsv')

    # Apply colormap with reversed normalization for blue to red transition
    colormapped_im = cmap(1 - normalized_depth_map)

    # Convert colormap to RGB values
    colormapped_im = (colormapped_im[:, :, :3] * 255).astype(np.uint8)

    return colormapped_im