import cv2
import numpy as np

def detect_text_in_manual_region(image_path, region, output_path="output.png",
                                 min_area=30, max_area=5000, delta=5,
                                 max_char_dist_x_ratio=1.5, max_char_dist_y_ratio=0.6,
                                 max_height_diff_ratio=0.3, curve_degree=2, min_chars_in_line=3):
    """
    Detect whether the text in a manually specified region is curved or straight.

    Args:
        image_path (str): Path to input image.
        region (tuple): (x, y, w, h) specifying the region to analyze.
        output_path (str): Path to save output image.
        Other parameters: tuning MSER and grouping.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image.")
        return

    x_roi, y_roi, w_roi, h_roi = region
    roi = img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create()
    mser.setDelta(delta)
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)

    regions, bboxes = mser.detectRegions(gray_roi)

    candidate_chars = []
    for i, box in enumerate(bboxes):
        x, y, w, h = box
        if w < 5 or h < 5 or w > 500 or h > 500:
            continue

        aspect_ratio = w / float(h)
        if not (0.1 < aspect_ratio < 10):
            continue

        if len(regions[i]) < 5:
            continue

        contour = regions[i]
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = float(area) / hull_area
        extent = float(area) / (w * h)

        if solidity < 0.3 or extent < 0.15:
            continue

        cx = x + w // 2
        cy = y + h // 2
        candidate_chars.append({"centroid": (cx, cy), "w": w, "h": h, "bbox": (x, y, w, h)})

    # Draw MSER boxes (optional)
    output = roi.copy()
    for c in candidate_chars:
        x, y, w, h = c['bbox']
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

    candidate_chars.sort(key=lambda item: item['centroid'][0])

    lines = []
    used = set()
    for i, c1