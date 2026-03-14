import numpy as np
from scipy.ndimage import gaussian_filter


def generate_density_map(image, points, sigma=4):

    h, w = image.shape[:2]

    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    for p in points:

        x = min(w - 1, max(0, int(p[0])))
        y = min(h - 1, max(0, int(p[1])))

        density[y, x] = 1

    density = gaussian_filter(density, sigma=sigma)

    return density