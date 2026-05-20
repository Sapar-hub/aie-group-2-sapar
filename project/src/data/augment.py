# src/data/augment.py

import cv2
import numpy as np
import random

def apply_augmentations(image: np.ndarray, scale_range: tuple = (0.7, 1.0)) -> np.ndarray:
    """
    Applies a series of random augmentations to an image patch.
    Used for transforming stamps in synthetic generation.

    Augmentations include:
    - Horizontal Flip
    - Rotation (small angles)
    - Scale with padding to preserve original size
    - Brightness/Contrast

    Args:
        image (np.ndarray): The input image patch.
        scale_range: Tuple of (min, max) scale factors. When scale < 1, padding is added.

    Returns:
        np.ndarray: The augmented image patch.
    """
    augmented_image = image.copy()
    h, w, _ = augmented_image.shape

    # 1. Horizontal Flip (50% chance)
    if random.random() < 0.5:
        augmented_image = cv2.flip(augmented_image, 1)

    # 2. Rotation (up to +/- 5 degrees)
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    augmented_image = cv2.warpAffine(augmented_image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 3. Scale with padding
    scale = random.uniform(scale_range[0], scale_range[1])
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(augmented_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if scale < 1.0:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        augmented_image = cv2.copyMakeBorder(
            scaled, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
            borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    else:
        augmented_image = scaled

    # 4. Brightness/Contrast
    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-15, 15)
    augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=beta)

    return augmented_image