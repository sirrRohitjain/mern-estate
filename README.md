import cv2
import numpy as np

def analyze_text_region(image_path, region_coords, output_image_path="output_curved_or_straight.png",
                        min_area=30, max_area=5000, max_variation=0.25, min_diversity=0.2, delta=5,
                        curve_degree=2, curvature_threshold=1.5):
    """
    Analyze a manually specified text region to determine if the text is straight or curved.

    Args:
        image_path (str): Path to the input image.
        region_coords (tuple): (x, y, w, h) coordinates of the region to analyze.
        output_image_path (str): Output image path to save results.
        min_area (int): Minimum MSER region area.
        max_area (int): Maximum MSER region area.
        max_variation (float): Maximum variation in MSER.
        min_diversity (float): Minimum diversity in MSER.
        delta (int): MSER delta parameter.
        curve_degree (int): Degree of polynomial to fit (e.g., 2 for quadratic).
        curvature_threshold (float): Threshold to classify as curved.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x, y, w, h = region_coords
    roi = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w].copy()

    # MSER detection
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setMaxVariation(max_variation)
    mser.setMinDiversity(min_diversity)
    mser.setDelta(delta)

    regions, bboxes = mser.detectRegions(roi)

    centroids = []
    for i, bbox in enumerate(bboxes):
        rx, ry, rw, rh = bbox
        if rw < 5 or rh < 5 or rw > 500 or rh > 500:
            continue
        aspect_ratio = rw / float(rh)
        if not (0.05 < aspect_ratio < 15):
            continue
        if len(regions[i]) < 5:
            continue

        contour = regions[i]
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        extent = float(area) / (rw * rh)
        if not (solidity > 0.3 and extent > 0.15):
            continue

        cx, cy = rx + rw // 2, ry + rh // 2
        centroids.append((cx, cy))

    centroids.sort(key=lambda p: p[0])
    if len(centroids) < 2:
        print("Not enough character points to analyze.")
        return

    x_coords = np.array([p[0] for p in centroids])
    y_coords = np.array([p[1] for p in centroids])

    try:
        coeffs = np.polyfit(x_coords, y_coords, deg=curve_degree)
        poly = np.poly1d(coeffs)

        poly_der2 = np.polyder(poly, 2)
        curvatures = np.abs(poly_der2(x_coords))
        mean_curvature = np.mean(curvatures)

        is_curved = mean_curvature > curvature_threshold

        x_fit = np.linspace(min(x_coords), max(x_coords), 100)
        y_fit = poly(x_fit)
        pts = np.array([[int(x), int(y)] for x, y in zip(x_fit, y_fit)], dtype=np.int32)

        pts[:, 0] += x
        pts[:, 1] += y
        color = (0, 0, 255) if is_curved else (0, 255, 0)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, 2)

        msg = "Curved Text" if is_curved else "Straight Text"
        cv2.putText(img, msg, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    except np.linalg.LinAlgError:
        print("Polynomial fitting failed.")
        return

    cv2.imwrite(output_image_path, img)
    print(f"Saved output with analysis to: {output_image_path}")

# ------------------ Main Execution ------------------

if __name__ == "__main__":
    input_image = "testt.png"            # Input image path
    output_image = "analyzed_output.png" # Output image path

    # Manually provide a region (x, y, w, h) to analyze
    region = (50, 100, 300, 80)

    analyze_text_region(
        image_path=input_image,
        region_coords=region,
        output_image_path=output_image,
        curve_degree=2,              # Degree of curve to fit
        curvature_threshold=1.5      # Threshold to decide if text is curved
    )