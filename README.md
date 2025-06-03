import cv2
import numpy as np
import easyocr

def detect_text_and_fit_curves(image_path, output_path="output.png", curve_degree=2):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    img_draw = image.copy()

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image)

    # Initialize MSER
    mser = cv2.MSER_create(_delta=5, _min_area=30, _max_area=5000)

    for (bbox, text, conf) in results:
        # Draw the EasyOCR word-level bounding box (green)
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(img_draw, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Get rectangular ROI around the word
        x_min = int(min(p[0] for p in bbox))
        y_min = int(min(p[1] for p in bbox))
        x_max = int(max(p[0] for p in bbox))
        y_max = int(max(p[1] for p in bbox))
        roi = image[y_min:y_max, x_min:x_max]

        # Convert ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply MSER to the grayscale ROI
        regions, _ = mser.detectRegions(gray_roi)

        centroids = []

        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)

            # Translate box coordinates back to full image
            box[:, 0] += x_min
            box[:, 1] += y_min

            # Draw MSER-based rotated rectangle (blue)
            cv2.polylines(img_draw, [box], isClosed=True, color=(255, 0, 0), thickness=1)

            # Compute centroid
            M = cv2.moments(box)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        # Fit curve if there are enough points
        if len(centroids) >= curve_degree + 1:
            x = np.array([pt[0] for pt in centroids])
            y = np.array([pt[1] for pt in centroids])

            try:
                poly = np.poly1d(np.polyfit(x, y, deg=curve_degree))
                x_curve = np.linspace(min(x), max(x), num=100)
                y_curve = poly(x_curve)
                curve_pts = np.array(list(zip(x_curve, y_curve)), dtype=np.int32)
                cv2.polylines(img_draw, [curve_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
            except np.linalg.LinAlgError:
                print("Curve fitting failed for one word (singular matrix).")

    # Save final image
    cv2.imwrite(output_path, img_draw)
    print(f"[\u2713] Saved result image to: {output_path}")

# Example usage
if __name__ == "__main__":
    detect_text_and_fit_curves("your_image.jpg", "result.png", curve_degree=2)


✅ The complete working code is now placed in the canvas above.

How it works:

Uses EasyOCR to get word-level bounding boxes.

Applies MSER inside each word box to extract character-like regions.

Draws parallelogram-like MSER boxes for each character.

Computes centroids of these boxes.

Fits and draws a polynomial curve through centroids to approximate text line curvature.


Let me know if you want to tweak filtering, improve character detection, or draw Bezier splines instead of polynomial curves.

