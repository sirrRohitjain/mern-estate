import cv2
import numpy as np
import os

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
                polyfit_degree_options=(1, 2, 3),
                max_dist=10):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image. Check the path.")

    x_roi, y_roi, w_roi, h_roi = roi_bbox
    roi_color = img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()
    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    preprocess_debug_name = os.path.splitext(output_image_path)[0] + "_grayscale.png"
    roi_processed = preprocess(roi_gray, save_debug=True, debug_name=preprocess_debug_name)

    # Binarization + Morphological operations
    _, thresh = cv2.threshold(roi_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8)

    valid_centroids = []
    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[i]
        if area < 30 or area > 5000:  # Filter by area
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.2 or aspect_ratio > 2.5:
            continue
        cx, cy = centroids[i]
        valid_centroids.append((cx, cy))
        cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.circle(roi_color, (int(cx), int(cy)), 2, (0, 0, 255), -1)

    if len(valid_centroids) < 3:
        print("Not enough centroids for curve fitting.")
        img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = roi_color
        cv2.imwrite(output_image_path, img)
        return

    # Sort by x-coordinate
    valid_centroids.sort(key=lambda pt: pt[0])
    cx = np.array([pt[0] for pt in valid_centroids])
    cy = np.array([pt[1] for pt in valid_centroids])

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
            covered = count(x_curve, y_curve, valid_centroids, max_dist)
            if covered > best_covered:
                best_curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)])
                best_covered = covered
                best_degree = degree
        except Exception:
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
    image_file = "./image_testing/image079.jpg"
    roi_box = (930, 857, 347, 124)
    detect_text(image_file, roi_box, output_image_path="./image_results/image079.jpg")