import cv2
import numpy as np
import easyocr
import math

# Parameters
image_path = "your_image.jpg"  # Replace with your image file
output_path = "output_curve_result.png"
min_mser_area = 30
max_mser_area = 2000
curve_degree = 2

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Read image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Image not found at path: {image_path}")

output_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get OCR word-level boxes
ocr_results = reader.readtext(image)

# Draw OCR word-level bounding boxes
for bbox, text, conf in ocr_results:
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(output_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  # Green box for word

# Initialize MSER
mser = cv2.MSER_create(_min_area=min_mser_area, _max_area=max_mser_area)
regions, _ = mser.detectRegions(gray)

# For each OCR box, detect MSER characters within it
for bbox, text, conf in ocr_results:
    box_pts = np.array(bbox, dtype=np.int32)
    x_min = np.min(box_pts[:, 0])
    x_max = np.max(box_pts[:, 0])
    y_min = np.min(box_pts[:, 1])
    y_max = np.max(box_pts[:, 1])

    word_roi_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(word_roi_mask, [box_pts], 255)

    char_centroids = []

    for region in regions:
        region_pts = np.array(region).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(region_pts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Filter by intersection with word-level OCR box
        inside_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(inside_mask, [box], -1, 255, -1)
        intersect = cv2.bitwise_and(word_roi_mask, inside_mask)
        if cv2.countNonZero(intersect) == 0:
            continue

        # Draw rotated rect (parallelogram-style)
        cv2.polylines(output_img, [box], True, (255, 0, 255), 1)  # Magenta box for character

        # Compute centroid
        M = cv2.moments(box)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            char_centroids.append((cX, cY))

    # Fit curve if enough characters
    if len(char_centroids) >= curve_degree + 1:
        x_vals = np.array([pt[0] for pt in char_centroids])
        y_vals = np.array([pt[1] for pt in char_centroids])

        try:
            poly_coeffs = np.polyfit(x_vals, y_vals, deg=curve_degree)
            poly_func = np.poly1d(poly_coeffs)

            x_line = np.linspace(min(x_vals), max(x_vals), 100)
            y_line = poly_func(x_line)

            curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_line, y_line)], dtype=np.int32)
            cv2.polylines(output_img, [curve_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)  # Red curve

        except np.linalg.LinAlgError:
            continue  # Skip if curve fitting fails

# Save output
cv2.imwrite(output_path, output_img)
print(f"Result saved to {output_path}")


✅ Here's the complete code that:

1. Uses EasyOCR to get word-level text boxes.


2. Uses MSER to detect character regions inside each word.


3. Draws rotated bounding boxes (parallelograms) for each character.


4. Computes centroids of these character boxes.


5. Fits a curve through the centroids for each word.


6. Saves the result to an output image.



Let me know if you'd like enhancements like noise filtering, adaptive MSER tuning, or improved curve fitting.

