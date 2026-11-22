"""Board bounding-box inference — numpy/cv2/onnxruntime only (no torch)."""

import cv2
import numpy as np

from src import consts


def _order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into TL, TR, BR, BL order."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([
        pts[s.argmin()],
        pts[d.argmin()],
        pts[s.argmax()],
        pts[d.argmax()],
    ], dtype=np.float32)


def _contour_to_quad(contour) -> np.ndarray | None:
    """Fit a 4-corner quad to a contour. Returns [4,2] float32 array or None."""
    peri = cv2.arcLength(contour, True)
    min_eps, max_eps = 0.0, 0.2
    best_approx = None
    for _ in range(20):
        eps = ((min_eps + max_eps) / 2.0) * peri
        approx = cv2.approxPolyDP(contour, eps, True)
        if len(approx) == 4:
            best_approx = approx
            break
        elif len(approx) > 4:
            min_eps = (min_eps + max_eps) / 2.0
        else:
            max_eps = (min_eps + max_eps) / 2.0
    if best_approx is None:
        rect = cv2.minAreaRect(contour)
        best_approx = cv2.boxPoints(rect)
    return _order_corners_tl_tr_br_bl(best_approx.reshape(4, 2))


def mask_to_corners(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray | None:
    """Extract 4 quad corners from the largest contour in a [H,W] mask.

    Returns [4,2] float32 array in TL/TR/BR/BL order, or None if mask is empty.
    """
    mask_np = (mask > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return _contour_to_quad(max(contours, key=cv2.contourArea))


def get_all_quads(
    mask: np.ndarray,
    min_area_fraction: float = 0.02,
    mask_threshold: float = 0.2,
) -> list[np.ndarray]:
    """Return ALL detected board regions from a segmentation mask.

    mask: [H, W] sigmoid values (output of get_bbox_mask).
    mask_threshold=0.2 catches low-confidence regions on multi-diagram pages.
    min_area_fraction: minimum contour area as fraction of mask area.

    Returns list of [4,2] corner arrays sorted top-to-bottom, may be empty.
    """
    mask_np = (mask > mask_threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    total_area = mask_np.shape[0] * mask_np.shape[1]
    min_area = total_area * min_area_fraction

    quads = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            break
        quad = _contour_to_quad(contour)
        if quad is not None:
            quads.append(quad)

    return quads
