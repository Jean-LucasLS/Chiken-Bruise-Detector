"""
Bruise Detection Module
=======================
HSV-based color segmentation pipeline for detecting bruise regions
on chicken carcasses in video frames.

Approach:
    Red/purple hues (H ∈ [0,10] ∪ [160,180]) are thresholded in HSV space,
    morphologically dilated to merge nearby regions, and enclosed in
    bounding-box annotations.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Region-of-interest crop ratios (relative to frame dimensions)
# ---------------------------------------------------------------------------
ROI_X_START: float = 0.25
ROI_X_END:   float = 0.75
ROI_Y_START: float = 0.20
ROI_Y_END:   float = 0.80

# ---------------------------------------------------------------------------
# HSV thresholds for bruise-associated red/purple hues
# ---------------------------------------------------------------------------
LOWER_HUE_1 = np.array([0,   50,  50],  dtype=np.uint8)
UPPER_HUE_1 = np.array([10,  255, 255], dtype=np.uint8)
LOWER_HUE_2 = np.array([160, 50,  50],  dtype=np.uint8)
UPPER_HUE_2 = np.array([180, 255, 255], dtype=np.uint8)

# Dilation kernel size — enlarges detected regions to capture bruise borders
DILATION_KERNEL_SIZE: tuple[int, int] = (50, 50)

# Bounding-box annotation color (BGR) and line thickness
BBOX_COLOR:     tuple[int, int, int] = (0, 255, 0)
BBOX_THICKNESS: int = 2


def crop_roi(frame: np.ndarray) -> np.ndarray:
    """Return the central region of interest cropped from *frame*."""
    height, width = frame.shape[:2]
    x0 = int(width  * ROI_X_START)
    x1 = int(width  * ROI_X_END)
    y0 = int(height * ROI_Y_START)
    y1 = int(height * ROI_Y_END)
    return frame[y0:y1, x0:x1]


def draw_bounding_boxes(frame: np.ndarray, contours: list) -> np.ndarray:
    """Overlay green bounding rectangles around each detected contour."""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), BBOX_COLOR, BBOX_THICKNESS)
    return frame


def detect_bruises(frame: np.ndarray) -> np.ndarray:
    """
    Detect bruise regions in a BGR video frame and return an annotated crop.

    Pipeline:
        1. Crop the region of interest from the raw frame.
        2. Convert the crop to HSV color space.
        3. Threshold red/purple hues associated with bruising.
        4. Dilate the binary mask to merge nearby blobs.
        5. Extract contours and draw bounding boxes on the BGR crop.

    Args:
        frame: Full BGR image as a NumPy array (H × W × 3).

    Returns:
        Annotated BGR crop (ROI only) with bounding boxes drawn over
        detected bruise regions.
    """
    roi  = crop_roi(frame)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_HUE_1, UPPER_HUE_1)
    mask2 = cv2.inRange(hsv, LOWER_HUE_2, UPPER_HUE_2)
    mask  = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones(DILATION_KERNEL_SIZE, np.uint8)
    mask   = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return draw_bounding_boxes(roi, contours)
