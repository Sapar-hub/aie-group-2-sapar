import cv2
import numpy as np
import random


def apply_augmentations(image: np.ndarray) -> np.ndarray:
    augmented_image = image.copy()
    h, w, _ = augmented_image.shape

    if random.random() < 0.5:
        augmented_image = cv2.flip(augmented_image, 1)

    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    augmented_image = cv2.warpAffine(augmented_image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    scale = random.uniform(0.8, 1.2)
    augmented_image = cv2.resize(augmented_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=beta)

    if random.random() < 0.5:
        cutout_h = int(h * random.uniform(0.1, 0.3))
        cutout_w = int(w * random.uniform(0.1, 0.3))
        if h > cutout_h and w > cutout_w:
            y1 = random.randint(0, h - cutout_h)
            x1 = random.randint(0, w - cutout_w)
            fill_color = random.randint(100, 200)
            augmented_image[y1:y1+cutout_h, x1:x1+cutout_w] = fill_color

    return augmented_image