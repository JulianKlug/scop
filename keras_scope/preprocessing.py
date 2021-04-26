from scipy import ndimage


def min_max_normalize(volume, min=-1000, max=400):
    """Normalize the volume"""
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img, desired_width, desired_height, desired_depth):
    """Resize"""
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    # Resize
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img