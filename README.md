

import cv2
import numpy as np
import os
from itertools import combinations

def preprocess(roi_gray, save_debug=False, debug_name="roi_grayscale.png"):
    if save_debug:
        cv2.imwrite(debug_name, roi_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi_gray)
    roi_blur = cv2.GaussianBlur(roi_eq, (3, 3), 0)
    return roi_blur

def count(x_curve, y_curve, centroids, max_dist=10):
    count = 0
    for cx, cy in centroids:
        for xc, yc in zip(x_curve, y_curve):
            if np.hypot(cx - xc, cy - yc) < max_dist:
                count += 1
                break
    return count

def detect_text(image_path, roi_bbox,
                                   output_image_path="output_text_shape.png",
                                   min_area=30, max_area=10000, delta=5,
                                   polyfit_degree_options=(1, 2, 3), max_dist=10):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image. Check the path.")

    x_roi, y_roi, w_roi, h_roi = roi_bbox
    roi_color = img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()
    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    preprocess_debug_name = os.path.splitext(output_image_path)[0] + "_grayscale.png"
    roi_processed = preprocess(roi_gray, save_debug=True, debug_name=preprocess_debug_name)

    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setDelta(delta)

    regions, bboxes = mser.detectRegions(roi_processed)

    char_centroids = []
    for box in bboxes:
        x, y, w, h = box
        if w < 5 or h < 5 or w > 500 or h > 500:
            continue
        cx = x + w // 2
        cy = y + h // 2
        char_centroids.append((float(cx), float(cy)))
        cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.circle(roi_color, (int(cx), int(cy)), 2, (0, 0, 255), -1)

    if len(char_centroids) < 3:
        print("Not enough centroids for curve fitting.")
        img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = roi_color
        cv2.imwrite(output_image_path, img)
        return

    centroids_sorted = sorted(char_centroids, key=lambda pt: pt[0])
    cx = np.array([pt[0] for pt in centroids_sorted])
    cy = np.array([pt[1] for pt in centroids_sorted])

    best_curve_pts = None
    best_covered = -1
    best_degree = None

    for degree in polyfit_degree_options:
        if len(cx) <= degree:
            continue
        try:
            coeffs = np.polyfit(cx, cy, degree)
            poly = np.poly1d(coeffs)
            x_curve = np.linspace(min(cx), max(cx), 200)
            y_curve = poly(x_curve)
            covered = count(x_curve, y_curve, centroids_sorted, max_dist)
            if covered > best_covered:
                best_curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)])
                best_covered = covered
                best_degree = degree
        except Exception as e:
            continue

    if best_curve_pts is not None:
        cv2.polylines(roi_color, [best_curve_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
        shape_type = f"CURVED (degree={best_degree}, coverage={best_covered})"
    else:
        pt1 = (int(cx[0]), int(cy[0]))
        pt2 = (int(cx[-1]), int(cy[-1]))
        cv2.line(roi_color, pt1, pt2, (255, 0, 0), 2)
        shape_type = "STRAIGHT (fallback)"

    img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = roi_color
    cv2.imwrite(output_image_path, img)
    print(f"Detected text shape: {shape_type}")
    print(f"Saved output to: {output_image_path}")
    print(f"Grayscale ROI saved to: {preprocess_debug_name}")

# Example usage
if __name__ == "__main__":
    image_file = "./image_testing/image079.jpg"  # image file name
    roi_box = (930,857,347,124)   # image dimensions(x,y,w,h)
    detect_text(image_file, roi_box, output_image_path="./image_results/image079.jpg")



    https://ieeexplore.ieee.org/abstract/document/7280160
