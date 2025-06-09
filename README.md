import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os

def preprocess_roi(roi_gray, save_debug=False, debug_name="roi_grayscale.png"):
    if save_debug:
        cv2.imwrite(debug_name, roi_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi_gray)
    roi_blur = cv2.GaussianBlur(roi_eq, (3, 3), 0)
    return roi_blur

def detect_text_curve_dbscan(image_path, roi_bbox,
                              output_image_path="output_text_shape.png",
                              min_area=30, max_area=10000, delta=5,
                              dbscan_eps=40, dbscan_min_samples=3,
                              polyfit_degree=2):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image. Check the path.")

    x_roi, y_roi, w_roi, h_roi = roi_bbox
    roi_color = img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()
    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    preprocess_debug_name = os.path.splitext(output_image_path)[0] + "_grayscale.png"
    roi_processed = preprocess_roi(roi_gray, save_debug=True, debug_name=preprocess_debug_name)

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

    if len(char_centroids) < 2:
        print(f"Only {len(char_centroids)} characters found — skipping curve fitting.")
        img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = roi_color
        cv2.imwrite(output_image_path, img)
        print(f"Output saved to: {output_image_path}")
        print(f"Grayscale ROI saved to: {preprocess_debug_name}")
        return

    # Apply DBSCAN clustering to centroids
    centroid_array = np.array(char_centroids)
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels = dbscan.fit_predict(centroid_array)

    # Find largest cluster (most points)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        print("No valid DBSCAN clusters found — fallback to straight line.")
        pt1 = (int(centroid_array[0][0]), int(centroid_array[0][1]))
        pt2 = (int(centroid_array[-1][0]), int(centroid_array[-1][1]))
        cv2.line(roi_color, pt1, pt2, (255, 0, 0), 2)
        shape_type = "STRAIGHT (no cluster)"
    else:
        best_label = unique_labels[np.argmax(counts)]
        inliers = centroid_array[labels == best_label]

        if len(inliers) >= 3:
            # Fit polynomial curve
            inliers = inliers[np.argsort(inliers[:, 0])]
            x_in = inliers[:, 0]
            y_in = inliers[:, 1]
            coeffs = np.polyfit(x_in, y_in, deg=polyfit_degree)
            poly = np.poly1d(coeffs)

            x_curve = np.linspace(min(x_in), max(x_in), 200)
            y_curve = poly(x_curve)
            curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)])
            cv2.polylines(roi_color, [curve_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
            shape_type = f"CURVED (DBSCAN cluster size={len(inliers)})"
        else:
            pt1 = (int(inliers[0][0]), int(inliers[0][1]))
            pt2 = (int(inliers[-1][0]), int(inliers[-1][1]))
            cv2.line(roi_color, pt1, pt2, (255, 0, 0), 2)
            shape_type = "STRAIGHT (cluster too small)"

    img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = roi_color
    cv2.imwrite(output_image_path, img)
    print(f"Detected text shape: {shape_type}")
    print(f"Saved output to: {output_image_path}")
    print(f"Grayscale ROI saved to: {preprocess_debug_name}")


# Example usage
if __name__ == "__main__":
    image_file = "curve1.jpg"  # Replace with your image file
    roi_box = (228, 130, 143, 43)  # Replace with your ROI (x, y, w, h)
    detect_text_curve_dbscan(image_file, roi_box, output_image_path="output_detected_curve.png")


✅ The complete code has been provided above. It includes:

Preprocessing (grayscale + CLAHE + blur),

MSER character detection,

Centroid extraction and visualization,

DBSCAN clustering to find coherent text line,

Polynomial curve fitting using numpy.polyfit,

Fallback to line if needed.


Let me know if you want to adapt this for multiple ROIs or enable visual debugging for each step.

