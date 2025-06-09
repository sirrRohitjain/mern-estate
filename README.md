import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import os

def preprocess_roi(roi_gray, save_debug=False, debug_name="roi_grayscale.png"):
    if save_debug:
        cv2.imwrite(debug_name, roi_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi_gray)
    roi_blur = cv2.GaussianBlur(roi_eq, (3, 3), 0)
    return roi_blur


def detect_text_curve(image_path, roi_bbox,
                      output_image_path="output_text_shape.png",
                      min_area=30, max_area=10000, delta=5,
                      min_chars_in_line=4):
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
        cv2.circle(roi_color, (int(cx), int(cy)), 2, (0, 0, 255), -1)  # Optional: show centroid

    if len(char_centroids) < min_chars_in_line:
        print("Not enough characters for curve fitting.")
        img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = roi_color
        cv2.imwrite(output_image_path, img)
        return

    # Sort left-to-right
    char_centroids.sort(key=lambda pt: pt[0])
    x_coords, y_coords = zip(*char_centroids)
    x_coords = np.array(x_coords, dtype=np.float64)
    y_coords = np.array(y_coords, dtype=np.float64)

    try:
        # Always try to fit B-spline curve
        tck, u = splprep([x_coords, y_coords], s=5, k=min(3, len(x_coords) - 1))
        u_new = np.linspace(0, 1, 200)
        x_spline, y_spline = splev(u_new, tck)

        curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_spline, y_spline)])
        cv2.polylines(roi_color, [curve_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
        shape_type = "CURVED"
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        pt1 = (int(x_coords[0]), int(y_coords[0]))
        pt2 = (int(x_coords[-1]), int(y_coords[-1]))
        cv2.line(roi_color, pt1, pt2, (255, 0, 0), 2)
        shape_type = "STRAIGHT (fallback)"

    img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = roi_color
    cv2.imwrite(output_image_path, img)

    print(f"Text shape detected: {shape_type}")
    print(f"Output saved to: {output_image_path}")
    print(f"Grayscale ROI saved to: {preprocess_debug_name}")


# -------- Example usage -----------
if __name__ == "__main__":
    image_file = "curve1.jpg"  # Replace with your image
    roi_box = (228, 130, 143, 43)  # (x, y, w, h)

    detect_text_curve(image_file, roi_box, output_image_path="output_detected_curve.png")