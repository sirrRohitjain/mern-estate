import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def detect_text_curve_or_line(image_path, roi_bbox, output_image_path="output_text_shape.png",
                              min_area=30, max_area=10000, delta=5, min_chars_in_line=4,
                              curvature_threshold=1.0):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image. Check the path.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_roi, y_roi, w_roi, h_roi = roi_bbox
    roi = gray[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    roi_color = img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi].copy()

    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setDelta(delta)

    regions, bboxes = mser.detectRegions(roi)

    char_centroids = []

    for i, box in enumerate(bboxes):
        x, y, w, h = box
        if w < 5 or h < 5 or w > 500 or h > 500:
            continue
        cx = x + w // 2
        cy = y + h // 2
        char_centroids.append((float(cx), float(cy)))  # Cast to float for splprep
        cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 1)

    if len(char_centroids) < min_chars_in_line:
        print("Not enough characters for curve or line fitting.")
        cv2.imwrite(output_image_path, img)
        return

    # Sort points left to right
    char_centroids.sort(key=lambda pt: pt[0])
    x_coords, y_coords = zip(*char_centroids)
    x_coords = np.array(x_coords, dtype=np.float64)
    y_coords = np.array(y_coords, dtype=np.float64)

    # Compute curvature to decide straight vs curved
    dy = np.gradient(y_coords)
    ddy = np.gradient(dy)
    curvature_score = np.std(ddy)

    try:
        if curvature_score > curvature_threshold:
            # CURVED (B-spline)
            tck, u = splprep([x_coords, y_coords], s=5, k=min(3, len(x_coords) - 1))
            u_new = np.linspace(0, 1, 200)
            x_spline, y_spline = splev(u_new, tck)
            curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_spline, y_spline)], dtype=np.int32)
            cv2.polylines(roi_color, [curve_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
            shape_type = "CURVED"
        else:
            # STRAIGHT
            pt1 = (int(x_coords[0]), int(y_coords[0]))
            pt2 = (int(x_coords[-1]), int(y_coords[-1]))
            cv2.line(roi_color, pt1, pt2, (255, 0, 0), 2)
            shape_type = "STRAIGHT"
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        # fallback to line
        pt1 = (int(x_coords[0]), int(y_coords[0]))
        pt2 = (int(x_coords[-1]), int(y_coords[-1]))
        cv2.line(roi_color, pt1, pt2, (255, 0, 0), 2)
        shape_type = "FALLBACK_LINE"

    # Paste the region back to the original image
    img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = roi_color
    cv2.imwrite(output_image_path, img)

    print(f"Detected text shape: {shape_type}")
    print(f"Saved output to: {output_image_path}")


# -------- Example Usage ----------
if __name__ == "__main__":
    image_file = "your_image.png"  # Replace with your file
    roi_box = (50, 100, 300, 80)   # Replace with your ROI (x, y, w, h)

    detect_text_curve_or_line(image_file, roi_box, output_image_path="output_detected_shape.png")