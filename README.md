'''import cv2
import numpy as np
import os
from itertools import combinations

def preprocess(roi_gray, save_debug=False, debug_name="roi_grayscale.png"):
    """Applies CLAHE and Gaussian Blur to the grayscale ROI."""
    if save_debug:
        cv2.imwrite(debug_name, roi_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi_gray)
    roi_blur = cv2.GaussianBlur(roi_eq, (3, 3), 0)
    return roi_blur

def count(x_curve, y_curve, centroids, max_dist=10):
    """Counts how many centroids are close to the given curve."""
    count = 0
    for cx, cy in centroids:
        # Find the minimum distance from the centroid to any point on the curve
        distances = np.hypot(cx - x_curve, cy - y_curve)
        if np.min(distances) < max_dist:
            count += 1
    return count

def filter_character_candidates(bboxes, roi_color_for_debug=None):
    """
    Filters MSER bounding boxes to keep only those that resemble characters.
    """
    if not bboxes:
        return []

    # --- Pass 1: Individual geometric properties ---
    candidates = []
    for x, y, w, h in bboxes:
        # Basic size filter
        if w < 5 or h < 5 or w > 100 or h > 100:
            if roi_color_for_debug is not None:
                cv2.rectangle(roi_color_for_debug, (x, y), (x + w, y + h), (0, 0, 128), 1) # Dark red for rejected
            continue

        # Aspect ratio filter
        aspect_ratio = h / w
        if aspect_ratio < 0.2 or aspect_ratio > 3.0:
            if roi_color_for_debug is not None:
                cv2.rectangle(roi_color_for_debug, (x, y), (x + w, y + h), (0, 0, 128), 1)
            continue
        
        candidates.append({'box': (x, y, w, h), 'cy': y + h / 2, 'h': h})

    if len(candidates) < 3:
        return []

    # --- Pass 2: Group properties (relative to other candidates) ---
    # Calculate median height and vertical position to find the main text line
    median_h = np.median([c['h'] for c in candidates])
    median_cy = np.median([c['cy'] for c in candidates])

    final_candidates = []
    for c in candidates:
        x, y, w, h = c['box']
        
        # Height consistency filter
        if h < median_h * 0.4 or h > median_h * 2.0:
            if roi_color_for_debug is not None:
                 cv2.rectangle(roi_color_for_debug, (x, y), (x + w, y + h), (128, 0, 128), 1) # Purple for rejected
            continue

        # Vertical alignment filter
        if abs(c['cy'] - median_cy) > median_h: # Allow deviation up to one median character height
            if roi_color_for_debug is not None:
                 cv2.rectangle(roi_color_for_debug, (x, y), (x + w, y + h), (128, 0, 128), 1)
            continue
            
        final_candidates.append(c['box'])
        
    return final_candidates


def detect_text(image_path, roi_bbox,
                output_image_path="output_text_shape.png",
                min_area=30, max_area=10000, delta=5,
                polyfit_degree_options=(1, 2, 3), max_dist=15):
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

    # Create a copy of the ROI for drawing debug rectangles
    roi_debug = roi_color.copy()

    # **CRITICAL STEP: Filter the bounding boxes**
    filtered_bboxes = filter_character_candidates(list(bboxes), roi_debug)

    char_centroids = []
    for box in filtered_bboxes:
        x, y, w, h = box
        cx = x + w // 2
        cy = y + h // 2
        char_centroids.append((float(cx), float(cy)))
        cv2.rectangle(roi_debug, (x, y), (x + w, y + h), (0, 255, 0), 1) # Green for accepted
        cv2.circle(roi_debug, (int(cx), int(cy)), 3, (255, 0, 255), -1) # Magenta centroid

    if len(char_centroids) < 3:
        print("Not enough filtered centroids for curve fitting.")
        img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = roi_debug # Show debug image
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
            print(f"Curve fitting failed for degree {degree}: {e}")
            continue

    if best_curve_pts is not None:
        cv2.polylines(roi_debug, [best_curve_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
        shape_type = f"CURVED (degree={best_degree}, coverage={best_covered}/{len(char_centroids)})"
    else:
        # Fallback to a straight line if curve fitting fails for all degrees
        if len(cx) > 1:
            pt1 = (int(cx[0]), int(cy[0]))
            pt2 = (int(cx[-1]), int(cy[-1]))
            cv2.line(roi_debug, pt1, pt2, (255, 0, 0), 2)
        shape_type = "STRAIGHT (fallback)"

    img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = roi_debug
    cv2.imwrite(output_image_path, img)
    print(f"Detected text shape: {shape_type}")
    print(f"Saved output to: {output_image_path}")
    print(f"Grayscale ROI saved to: {preprocess_debug_name}")


if __name__ == "__main__":
    image_file = "./image_testing/image040.jpg"  # image file name
    roi_box = (584,63,396,270)  # image dimensions(x,y,w,h)
    
    # Create results directory if it doesn't exist
    os.makedirs("./image_results", exist_ok=True)
    
    detect_text(image_file, roi_box, output_image_path="./image_results/image040_filtered.jpg")'''
import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN


def preprocess(roi_gray, save_debug=False, debug_name="roi_grayscale.png"):
    """Applies CLAHE and Gaussian Blur to the grayscale ROI."""
    if save_debug:
        cv2.imwrite(debug_name, roi_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi_gray)
    roi_blur = cv2.GaussianBlur(roi_eq, (3, 3), 0)
    return roi_blur

def filter_character_candidates(bboxes, roi_color_for_debug=None, roi_bbox=None):
    """
    Filters MSER bounding boxes to keep only those that resemble characters.
    """
    if not isinstance(bboxes, list):
        bboxes = bboxes.tolist()  # Convert to list if it's a NumPy array

    if not bboxes:
        return []

    x_roi, y_roi, w_roi, h_roi = roi_bbox if roi_bbox else (0, 0, 10000, 10000)

    # --- Pass 1: Individual geometric properties ---
    candidates = []
    for x, y, w, h in bboxes:
        # Basic size filter
        if w < 10 or h < 10 or w > 100 or h > 100:
            if roi_color_for_debug is not None:
                cv2.rectangle(roi_color_for_debug, (x, y), (x + w, y + h), (0, 0, 128), 1)  # Dark red for rejected
            continue

        # Aspect ratio filter
        aspect_ratio = h / w
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            if roi_color_for_debug is not None:
                cv2.rectangle(roi_color_for_debug, (x, y), (x + w, y + h), (0, 0, 128), 1)
            continue

        # Proximity to ROI borders filter
        if x < 10 or y < 10 or x + w > w_roi - 10 or y + h > h_roi - 10:
            if roi_color_for_debug is not None:
                cv2.rectangle(roi_color_for_debug, (x, y), (x + w, y + h), (0, 0, 128), 1)
            continue

        candidates.append({'box': (x, y, w, h), 'cy': y + h / 2, 'h': h})

    if len(candidates) < 3:
        return []

    # --- Pass 2: Group properties (relative to other candidates) ---
    # Calculate median height and vertical position to find the main text line
    median_h = np.median([c['h'] for c in candidates])
    median_cy = np.median([c['cy'] for c in candidates])

    final_candidates = []
    for c in candidates:
        x, y, w, h = c['box']

        # Height consistency filter
        if h < median_h * 0.5 or h > median_h * 2.0:
            if roi_color_for_debug is not None:
                cv2.rectangle(roi_color_for_debug, (x, y), (x + w, y + h), (128, 0, 128), 1)  # Purple for rejected
            continue

        # Vertical alignment filter
        if abs(c['cy'] - median_cy) > median_h * 0.75:  # Allow deviation up to 0.75 times the median character height
            if roi_color_for_debug is not None:
                cv2.rectangle(roi_color_for_debug, (x, y), (x + w, y + h), (128, 0, 128), 1)
            continue

        final_candidates.append(c['box'])

    return final_candidates

def fit_curve_to_centroids(centroids, max_dist=15):
    """Fit a curve to the centroids using DBSCAN clustering."""
    if len(centroids) < 3:
        return None

    # Convert centroids to numpy array
    centroids = np.array(centroids)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=max_dist, min_samples=2).fit(centroids)
    labels = clustering.labels_

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    if not unique_labels:
        return None

    # Fit a curve to the largest cluster
    largest_cluster_label = max(unique_labels, key=lambda label: np.sum(labels == label))
    cluster_points = centroids[labels == largest_cluster_label]

    # Fit a polynomial curve to the cluster points
    x = cluster_points[:, 0]
    y = cluster_points[:, 1]
    degree = min(3, len(cluster_points) - 1)
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    # Generate points on the fitted curve
    x_curve = np.linspace(min(x), max(x), 200)
    y_curve = poly(x_curve)

    return np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)])



def detect_text_and_fit_curve(image_path, roi_bbox, output_image_path="output_text_shape.png"):
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError("Could not load image. Check the path.")

    x_roi, y_roi, w_roi, h_roi = roi_bbox

    # Crop the ROI from the original image
    roi_color = img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()
    
    # Convert to grayscale and apply preprocessing
    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    preprocess_debug_name = os.path.splitext(output_image_path)[0] + "_grayscale.png"
    roi_processed = preprocess(roi_gray, save_debug=True, debug_name=preprocess_debug_name)

    mser = cv2.MSER_create()
    regions, bboxes = mser.detectRegions(roi_processed)

    # Draw bounding boxes around detected text blocks
    roi_debug = roi_color.copy()
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        cv2.rectangle(roi_debug, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green for accepted

    # Compute centroids of the filtered text regions
    char_centroids = []
    for box in bboxes:
        x, y, w, h = box
        cx = x + w // 2
        cy = y + h // 2
        char_centroids.append((cx, cy))
        cv2.circle(roi_debug, (int(cx), int(cy)), 3, (255, 0, 255), -1)  # Magenta centroid

    # Fit a curve to the centroids
    if len(char_centroids) > 1:
        curve_points = fit_curve_to_centroids(char_centroids)
        cv2.polylines(roi_debug, [curve_points.reshape(-1, 1, 2)], False, (0, 0, 255), 2)

    # Save the result
    img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = roi_debug
    cv2.imwrite(output_image_path, img)
    print(f"Saved output to: {output_image_path}")
    print(f"Grayscale ROI saved to: {preprocess_debug_name}")

if __name__ == "__main__":
    image_file = "image079.jpg"  # image file name
    roi_box = (930,857,347,124)  # image dimensions (x, y, w, h)
    
    # Create results directory if it doesn't exist
    os.makedirs("./image_results", exist_ok=True)
    
    detect_text_and_fit_curve(image_file, roi_box, output_image_path="testimage079.jpg")
