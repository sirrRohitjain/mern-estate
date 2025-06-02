import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def detect_text_curve_or_line(image_path, roi_bbox, output_image_path="output_text_shape.png",
                              min_area=30, max_area=10000, delta=5, min_chars_in_line=4,
                              curvature_threshold=1.0):
    """
    Detect characters inside a manually given bounding box (ROI),
    and fit either a curve (B-spline) or straight line depending on the shape.

    Args:
        image_path (str): Path to the input image.
        roi_bbox (tuple): Region of interest as (x, y, w, h).
        output_image_path (str): File to save output image with shape drawn.
        min_area (int): Minimum MSER region area.
        max_area (int): Maximum MSER region area.
        delta (int): MSER delta parameter.
        min_chars_in_line (int): Minimum number of detected characters.
        curvature_threshold (float): Threshold on curvature std dev to decide straight vs curved.
    """
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
        char_centroids.append((cx, cy))
        cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green box

    if len(char_centroids) < min_chars_in_line:
        print("Not enough characters for curve or line fitting.")
        cv2.imwrite(output_image_path, img)
        return

    # Sort by x to go left to right
    char_centroids.sort(key=lambda pt: pt[0])
    x_coords, y_coords = zip(*char_centroids)

    # Compute curvature to determine shape
    dy = np.gradient(y_coords)
    ddy = np.gradient(dy)
    curvature_score = np.std(ddy)

    if curvature_score > curvature_threshold:
        # CURVED - B-spline fit
        tck, u = splprep([x_coords, y_coords], s=10)
        unew = np.linspace(0, 1.0, num=200)
        out = splev(unew, tck)
        curve_points = np.array([[int(x), int(y)] for x, y in zip(out[0], out[1])])
        cv2.polylines(roi_color, [curve_points.reshape(-1, 1, 2)], False, (0, 0, 255), 2)  # Red
        shape_type = "CURVED"
    else:
        # STRAIGHT - simple line from start to end
        pt1 = (int(x_coords[0]), int(y_coords[0]))
        pt2 = (int(x_coords[-1]), int(y_coords[-1]))
        cv2.line(roi_color, pt1, pt2, (255, 0, 0), 2)  # Blue
        shape_type = "STRAIGHT"

    # Put ROI back into original image
    img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = roi_color
    cv2.imwrite(output_image_path, img)

    print(f"Text is detected as: {shape_type}")
    print(f"Output saved to: {output_image_path}")


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    image_file = "your_image.png"
    roi_box = (50, 100, 300, 80)  # Replace with manually identified coordinates

    detect_text_curve_or_line(image_file, roi_box, output_image_path="detected_text_shape.png")