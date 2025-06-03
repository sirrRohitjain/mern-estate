import cv2
import numpy as np
import easyocr

def fit_curve_through_points(points, degree=2):
    """Fits a polynomial curve through the given points."""
    if len(points) < degree + 1:
        return None
    x = [pt[0] for pt in points]
    y = [pt[1] for pt in points]
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    x_new = np.linspace(min(x), max(x), 100)
    y_new = poly(x_new)
    return np.array([[int(xi), int(yi)] for xi, yi in zip(x_new, y_new)], dtype=np.int32)

def draw_text_curve(img_path, output_path="output_curve_easyocr.png", degree=2):
    reader = easyocr.Reader(['en'], gpu=False)
    img = cv2.imread(img_path)
    img_draw = img.copy()
    h_img, w_img = img.shape[:2]

    results = reader.readtext(img)

    for (bbox, text, conf) in results:
        bbox_np = np.array(bbox, dtype=np.int32)
        x_min = max(int(min(p[0] for p in bbox)), 0)
        y_min = max(int(min(p[1] for p in bbox)), 0)
        x_max = min(int(max(p[0] for p in bbox)), w_img)
        y_max = min(int(max(p[1] for p in bbox)), h_img)

        cropped = img[y_min:y_max, x_min:x_max]
        char_results = reader.readtext(cropped, detail=1, paragraph=False)

        char_centroids = []
        for (cbbox, ctext, cconf) in char_results:
            c_pts = np.array(cbbox, dtype=np.int32)
            cx = int(np.mean(c_pts[:, 0])) + x_min
            cy = int(np.mean(c_pts[:, 1])) + y_min
            char_centroids.append((cx, cy))

        if len(char_centroids) >= degree + 1:
            curve_pts = fit_curve_through_points(char_centroids, degree=degree)
            if curve_pts is not None:
                cv2.polylines(img_draw, [curve_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)

        # Draw the word box
        cv2.polylines(img_draw, [bbox_np.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imwrite(output_path, img_draw)
    print(f"[✓] Saved output with curves to {output_path}")

# Example Usage
draw_text_curve("curve1.jpg", "output_curved_easyocr.png", degree=2)