import cv2
import easyocr
import numpy as np
import torch

def fit_curve_through_text(res, curve_degree=2, color=(0, 0, 255)):
    """Fit a curve through the center points of the bounding boxes."""
    if len(res) < curve_degree + 1:
        return None  # Not enough points to fit curve

    # Get center points of bounding boxes
    centers = []
    for bbox, _, _ in res:
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        center_x = sum(xs) / 4
        center_y = sum(ys) / 4
        centers.append((center_x, center_y))

    centers = sorted(centers, key=lambda x: x[0])  # sort left to right

    x_coords = np.array([pt[0] for pt in centers])
    y_coords = np.array([pt[1] for pt in centers])

    try:
        coeffs = np.polyfit(x_coords, y_coords, curve_degree)
        poly = np.poly1d(coeffs)

        x_new = np.linspace(x_coords.min(), x_coords.max(), 100)
        y_new = poly(x_new)
        points = np.array([[int(x), int(y)] for x, y in zip(x_new, y_new)], dtype=np.int32)
        return points
    except:
        return None

def detect_and_draw_easyocr(image_path, output_path="output_easyocr.png", draw_curve=True):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # Convert image to RGB for easyocr
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    # Detect text
    res = reader.readtext(img_rgb)

    # Draw bounding boxes
    for bbox, text, conf in res:
        bbox = np.array(bbox, dtype=np.int32)
        cv2.polylines(img, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)

    # Optionally draw a curve through detected text line
    if draw_curve:
        curve_pts = fit_curve_through_text(res, curve_degree=2)
        if curve_pts is not None:
            cv2.polylines(img, [curve_pts], isClosed=False, color=(0, 0, 255), thickness=2)

    # Save and display output
    cv2.imwrite(output_path, img)
    print(f"Output saved to: {output_path}")

# --- MAIN USAGE ---
if __name__ == "__main__":
    image_file = "your_image.jpg"  # <- change this
    detect_and_draw_easyocr(image_file, "output_easyocr_result.png")