import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# ----------------------------------------
# Preprocessing: CLAHE + Blur + Edges
# ----------------------------------------
def preprocess_roi(gray_roi):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_img = clahe.apply(gray_roi)
    blur_img = cv2.GaussianBlur(eq_img, (3, 3), 0)
    edges = cv2.Canny(blur_img, 50, 150)
    return edges

# ----------------------------------------
# Weighted Euclidean Distance
# ----------------------------------------
def weighted_distance(u, v, weights):
    return np.sqrt(np.sum(weights * (u - v) ** 2))

# ----------------------------------------
# Main Function: MSER + Clustering
# ----------------------------------------
def detect_text_with_hierarchical_clustering(image_path, roi_bbox,
                                              output_image_path="output.png",
                                              min_area=30,
                                              max_area=10000,
                                              distance_thresh=50,
                                              weights=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image.")

    x, y, w, h = roi_bbox
    roi = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    processed = preprocess_roi(gray)

    # MSER Detection
    mser = cv2.MSER_create(_min_area=min_area, _max_area=max_area)
    regions, boxes = mser.detectRegions(processed)

    char_boxes = []
    features = []
    for box in boxes:
        x_box, y_box, w_box, h_box = box
        if w_box < 5 or h_box < 5:
            continue
        aspect_ratio = w_box / float(h_box)
        if not (0.25 < aspect_ratio < 5.0):
            continue
        cx = x_box + w_box / 2
        cy = y_box + h_box / 2
        char_boxes.append((x_box, y_box, w_box, h_box))
        features.append([cx, cy, w_box, h_box])

    if len(features) < 3:
        print("Not enough MSER boxes for clustering.")
        return

    X = np.array(features)
    weights = np.array(weights or [1.0, 0.5, 0.2, 0.2])  # Default weights

    # Distance matrix using custom weighted distance
    dist_matrix = squareform(pdist(X, lambda u, v: weighted_distance(u, v, weights)))
    Z = linkage(dist_matrix, method='average')
    labels = fcluster(Z, t=distance_thresh, criterion='distance')

    output_roi = roi.copy()
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        indices = np.where(labels == lbl)[0]
        if len(indices) < 3:
            continue  # Skip small clusters

        # Draw bounding boxes
        for i in indices:
            x_box, y_box, w_box, h_box = char_boxes[i]
            cv2.rectangle(output_roi, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 255, 0), 1)

        # Fit a curve to centroids
        cluster_centroids = [(X[i][0], X[i][1]) for i in indices]
        cluster_centroids = sorted(cluster_centroids, key=lambda pt: pt[0])
        cx_vals = np.array([pt[0] for pt in cluster_centroids])
        cy_vals = np.array([pt[1] for pt in cluster_centroids])

        try:
            coeffs = np.polyfit(cx_vals, cy_vals, deg=2)  # Quadratic fit
            poly = np.poly1d(coeffs)
            x_curve = np.linspace(min(cx_vals), max(cx_vals), 200)
            y_curve = poly(x_curve)
            pts = np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)])
            cv2.polylines(output_roi, [pts.reshape(-1, 1, 2)], isClosed=False, color=(0, 0, 255), thickness=2)
        except Exception as e:
            print("Curve fitting failed for cluster:", e)

    img[y:y+h, x:x+w] = output_roi
    cv2.imwrite(output_image_path, img)
    print(f"[✔] Output saved to {output_image_path}")

# ----------------------------------------
# Example Usage
# ----------------------------------------
if __name__ == "__main__":
    detect_text_with_hierarchical_clustering(
        image_path="curve1.jpg",  # 🔁 Replace with your image file
        roi_bbox=(228, 130, 143, 43),  # 🔁 Replace with your ROI (x, y, w, h)
        output_image_path="output_detected_text.png",
        min_area=30,
        max_area=10000,
        distance_thresh=50,
        weights=[1.0, 0.5, 0.2, 0.2],  # Tunable: [cx, cy, w, h]
    )