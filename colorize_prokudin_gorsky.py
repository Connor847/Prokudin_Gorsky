"""
Prokudin-Gorsky Image Colorization
====================================
Aligns and combines three single-channel (B, G, R) images captured separately
into a single full-color RGB image.

Pipeline:
    1. Load a .tif file containing three stacked grayscale channels
    2. Crop white and black borders from each channel
    3. Align G and R channels to the B channel using a gradient-based
       normalized cross-correlation pyramid search
    4. Normalize channel brightness
    5. Remove edge artifacts introduced by alignment
    6. Save the result

Usage:
    Set INPUT_PATH and OUTPUT_PATH at the bottom of this file, then run:
        python colorize_prokudin_gorsky.py
"""

import numpy as np
import skimage.io as skio
from skimage.util import img_as_float
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Border detection & cropping
# ---------------------------------------------------------------------------

def row_is_white(pixels, y, width, threshold=0.94, samples=10):
    step = max(1, width // samples)
    return all(pixels[y][x] > threshold for x in range(0, width, step))


def col_is_white(pixels, x, height, threshold=0.94, samples=10):
    step = max(1, height // samples)
    return all(pixels[y, x] > threshold for y in range(0, height, step))


def crop_white_border(image):
    """Remove solid-white border rows/columns from a grayscale image."""
    height, width = image.shape

    top = 0
    while top < height and row_is_white(image, top, width):
        top += 1

    bottom = height - 1
    while bottom > top and row_is_white(image, bottom, width):
        bottom -= 1

    left = 0
    while left < width and col_is_white(image, left, height):
        left += 1

    right = width - 1
    while right > left and col_is_white(image, right, height):
        right -= 1

    return image[top:bottom + 1, left:right + 1]


def row_is_black(pixels, y, width, avg_threshold=0.2, var_threshold=0.1, samples=100):
    step = max(1, width // samples)
    sampled = [pixels[y, x] for x in range(0, width, step)]
    avg = sum(sampled) / len(sampled)
    variance = sum((v - avg) ** 2 for v in sampled) / len(sampled)
    return avg < avg_threshold and variance < var_threshold


def col_is_black(pixels, x, height, avg_threshold=0.2, var_threshold=0.1, samples=100):
    step = max(1, height // samples)
    sampled = [pixels[y, x] for y in range(0, height, step)]
    avg = sum(sampled) / len(sampled)
    variance = sum((v - avg) ** 2 for v in sampled) / len(sampled)
    return avg < avg_threshold and variance < var_threshold


def crop_black_border(image):
    """Remove near-black border rows/columns from a grayscale image."""
    height, width = image.shape

    top = 0
    while top < height and row_is_black(image, top, width):
        top += 1

    bottom = height - 1
    while bottom > top and row_is_black(image, bottom, width):
        bottom -= 1

    left = 0
    while left < width and col_is_black(image, left, height):
        left += 1

    right = width - 1
    while right > left and col_is_black(image, right, height):
        right -= 1

    print(f"  black crop -> top={top}, bottom={bottom}, left={left}, right={right}")
    return image[top:bottom + 1, left:right + 1]


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def simple_gradient(image):
    """Compute gradient magnitude using finite differences."""
    dy = np.diff(image, axis=0, prepend=0)
    dx = np.diff(image, axis=1, prepend=0)
    return np.sqrt(dx ** 2 + dy ** 2)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def build_pyramid(image, min_size=256):
    """Build a coarse-to-fine image pyramid by successive 2x downsampling."""
    pyramid = [image]
    while min(pyramid[-1].shape) > min_size:
        pyramid.append(pyramid[-1][::2, ::2])
    return pyramid


def pyramid_offset(image, reference, dx, dy, window_length=4):
    """
    Search for the best (dx, dy) offset within a local window at one pyramid level.

    Parameters
    ----------
    image, reference : 2-D arrays at the current pyramid level
    dx, dy           : starting offset (propagated from coarser level, already scaled)
    window_length    : half-width of the search window around the starting offset

    Returns
    -------
    (best_dx, best_dy) : integer pixel offsets
    """
    image_grad = simple_gradient(image)
    reference_grad = simple_gradient(reference)

    h, w = image.shape
    top, bottom = h // 5, 4 * h // 5
    left, right = w // 5, 4 * w // 5

    ref_crop = reference_grad[top:bottom, left:right]
    ref_norm = (ref_crop - np.mean(ref_crop)) / (np.std(ref_crop) + 1e-8)

    best_score = -np.inf
    best_dx, best_dy = dx, dy

    for ddy in range(-window_length, window_length):
        for ddx in range(-window_length, window_length):
            shifted = np.roll(image_grad, dy + ddy, axis=0)
            shifted = np.roll(shifted, dx + ddx, axis=1)
            img_crop = shifted[top:bottom, left:right]
            img_norm = (img_crop - np.mean(img_crop)) / (np.std(img_crop) + 1e-8)
            score = np.sum(img_norm * ref_norm)
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx + ddx, dy + ddy

    return best_dx, best_dy


def align_pyramid(image, reference):
    """
    Align *image* to *reference* using a coarse-to-fine pyramid NCC search.

    Returns
    -------
    aligned : 2-D array — shifted version of *image*
    dx, dy  : final pixel offsets applied
    """
    image_grad = simple_gradient(image)
    reference_grad = simple_gradient(reference)

    ref_pyramid = build_pyramid(reference_grad)
    im_pyramid = build_pyramid(image_grad)

    dx, dy = 0, 0
    for level in range(len(im_pyramid) - 1, -1, -1):
        dx, dy = pyramid_offset(im_pyramid[level], ref_pyramid[level], dx * 2, dy * 2, window_length=4)

    aligned = np.roll(np.roll(image, dy, axis=0), dx, axis=1)
    return aligned, dx, dy


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def normalize(channel):
    """Stretch channel values to [0, 1]."""
    lo, hi = np.min(channel), np.max(channel)
    return (channel - lo) / (hi - lo)


def row_is_misaligned(im_out, y, threshold=0.2, samples=100):
    h, w, _ = im_out.shape
    step = max(1, w // samples)
    diffs = [
        max(abs(im_out[y, x, 0] - im_out[y, x, 1]),
            abs(im_out[y, x, 1] - im_out[y, x, 2]),
            abs(im_out[y, x, 0] - im_out[y, x, 2]))
        for x in range(0, w, step)
    ]
    return sum(diffs) / len(diffs) > threshold


def col_is_misaligned(im_out, x, threshold=0.2, samples=100):
    h, w, _ = im_out.shape
    step = max(1, h // samples)
    diffs = [
        max(abs(im_out[y, x, 0] - im_out[y, x, 1]),
            abs(im_out[y, x, 1] - im_out[y, x, 2]),
            abs(im_out[y, x, 0] - im_out[y, x, 2]))
        for y in range(0, h, step)
    ]
    return sum(diffs) / len(diffs) > threshold


def remove_artifacts(im_out, threshold=0.35, samples=10):
    """Crop rows/columns with high inter-channel disagreement from the edges."""
    height, width, _ = im_out.shape

    top = 0
    while top < height and row_is_misaligned(im_out, top, threshold, samples):
        top += 1

    bottom = height - 1
    while bottom > top and row_is_misaligned(im_out, bottom, threshold, samples):
        bottom -= 1

    left = 0
    while left < width and col_is_misaligned(im_out, left, threshold, samples):
        left += 1

    right = width - 1
    while right > left and col_is_misaligned(im_out, right, threshold, samples):
        right -= 1

    return im_out[top:bottom + 1, left:right + 1, :]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_channels(path):
    """
    Load a stacked .tif and split into B, G, R single-channel float arrays.

    The file is assumed to contain three equal-height horizontal strips
    ordered top-to-bottom: Blue, Green, Red.
    """
    im = img_as_float(skio.imread(path))
    height = int(np.floor(im.shape[0] / 3.0))
    b = im[:height]
    g = im[height:2 * height]
    r = im[2 * height:]
    return b, g, r


def crop_channels(b, g, r):
    """Apply white then black border cropping to each channel and unify size."""
    channels = []
    for ch in (b, g, r):
        ch = crop_white_border(ch)
        ch = crop_black_border(ch)
        channels.append(ch)

    min_h = min(c.shape[0] for c in channels)
    min_w = min(c.shape[1] for c in channels)
    return tuple(c[:min_h, :min_w] for c in channels)


def build_color_image(input_path, output_path=None, show=False):
    """
    Full pipeline: load -> crop -> align -> normalize -> remove artifacts -> save.

    Parameters
    ----------
    input_path  : str  path to the stacked .tif input file
    output_path : str  path to save the result (e.g. 'out.jpg'); skipped if None
    show        : bool display the result with matplotlib
    """
    print(f"Loading: {input_path}")
    b, g, r = load_channels(input_path)

    print("Cropping borders...")
    b, g, r = crop_channels(b, g, r)

    print("Aligning G to B...")
    ag, g_dx, g_dy = align_pyramid(g, b)
    print(f"  G offset -> dx={g_dx}, dy={g_dy}")

    print("Aligning R to B...")
    ar, r_dx, r_dy = align_pyramid(r, b)
    print(f"  R offset -> dx={r_dx}, dy={r_dy}")

    r_ch = normalize(ar)
    g_ch = normalize(ag)
    b_ch = normalize(b)

    # Match channel brightness to a common average
    avg = (np.mean(r_ch) + np.mean(g_ch) + np.mean(b_ch)) / 3
    r_ch = np.clip(r_ch * (avg / np.mean(r_ch)), 0, 1)
    g_ch = np.clip(g_ch * (avg / np.mean(g_ch)), 0, 1)
    b_ch = np.clip(b_ch * (avg / np.mean(b_ch)), 0, 1)

    im_out = np.dstack([r_ch, g_ch, b_ch])

    print("Removing edge artifacts...")
    im_out = remove_artifacts(im_out, threshold=0.35, samples=10)

    if output_path:
        skio.imsave(output_path, (im_out * 255).astype(np.uint8))
        print(f"Saved to: {output_path}")

    if show:
        plt.imshow(im_out)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return im_out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    INPUT_PATH = "/Users/connorfeldman/Documents_Folder/College/MSDS/Computer_Vision_II/coms4732_hw1_data/emir.tif"   # <-- update this
    OUTPUT_PATH = "Emir.jpg"            # <-- update this (or set to None to skip saving)

    build_color_image(INPUT_PATH, OUTPUT_PATH, show=True)
