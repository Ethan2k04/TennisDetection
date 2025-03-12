import numpy as np


# will come from config.json
min_x = 80
min_y = 63 
max_x = 601
max_y = 403

def make_binary_bitmap(h: np.ndarray, s: np.ndarray, v: np.ndarray, min_hue: int, max_hue: int, min_value: int, max_value: int) -> np.ndarray:
    """
    Create a binary bitmap based on hue and value thresholds.

    Parameters:
    h (np.ndarray): Hue channel of the image.
    v (np.ndarray): Value channel of the image.
    min_hue (int): Minimum hue threshold.
    max_hue (int): Maximum hue threshold.
    min_value (int): Minimum value threshold.
    max_value (int): Maximum value threshold.

    Returns:
    np.ndarray: Binary bitmap.
    """
    binary_bitmap = np.ones_like(v)

    for (i, j), value in np.ndenumerate(v):
        if i < min_y or i > max_y :
            binary_bitmap[i,j] = 0
            continue 

        if j < min_x or j > max_x :
            binary_bitmap[i,j] = 0
            continue

        h_value = h[i, j]
        s_value = s[i,j]

        # white
        if s_value < 40 :
            binary_bitmap[i,j] = 0
            continue

        if h_value > max_hue or h_value < min_hue:
            binary_bitmap[i, j] = 0
        elif value > max_value or value < min_value:
            binary_bitmap[i, j] = 0

    return binary_bitmap