import cv2
import numpy as np
import easyocr
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import Polynomial
import os


def detect_and_analyze_text(image_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image)

    for bbox, text, conf in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Crop word from original image
        x_min = min([p[0] for p in bbox])
        x_max = max([p[0] for p in bbox])
        y_min = min([p[1] for p in bbox])
        y_max = max([p[1] for p in bbox])

        word_crop = orig_image[int(y_min):int(y_max), int(x_min):int(x_max)]

        if word_crop.size == 0:
            continue

        gray = cv2.cvtColor(word_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours (character blobs)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) + int(x_min)
                cy = int(M['m01'] / M['m00']) + int(y_min)
                centroids.append((cx, cy))
                cv2.drawContours(image, [box + [int(x_min), int(y_min)]], 0, (0, 0, 255), 1)

        if len(centroids) >= 3:
            centroids = sorted(centroids, key=lambda pt: pt[0])
            x_vals = np.array([pt[0] for pt in centroids])
            y_vals = np.array([pt[1] for pt in centroids])

            # Fit polynomial (degree 2 for curve detection)
            coeffs = Polynomial.fit(x_vals, y_vals, 2).convert().coef
            y_pred_curve = coeffs[0] + coeffs[1]*x_vals + coeffs[2]*x_vals**2

            # Fit straight line
            lin_model = LinearRegression().fit(x_vals.reshape(-1,1), y_vals)
            y_pred_line = lin_model.predict(x_vals.reshape(-1,1))

            # Compare error
            error_curve = np.mean((y_vals - y_pred_curve)**2)
            error_line = np.mean((y_vals - y_pred_line)**2)

            if error_curve < error_line * 0.8:
                # Draw curve
                for i in range(len(x_vals) - 1):
                    pt1 = (int(x_vals[i]), int(y_pred_curve[i]))
                    pt2 = (int(x_vals[i+1]), int(y_pred_curve[i+1]))
                    cv2.line(image, pt1, pt2, (0, 0, 255), 2)
            else:
                # Draw straight line
                pt1 = (int(x_vals[0]), int(y_pred_line[0]))
                pt2 = (int(x_vals[-1]), int(y_pred_line[-1]))
                cv2.line(image, pt1, pt2, (255, 0, 0), 2)

    # Save result
    cv2.imwrite(output_path, image)
    print(f"Saved result to {output_path}")


# Example usage:
if __name__ == '__main__':
    image_path = 'curve1.jpg'  # Replace with your image path
    output_path = 'output_result.jpg'
    detect_and_analyze_text(image_path, output_path)


✅ The complete and corrected code has been added to the canvas. This implementation:

1. Uses EasyOCR to detect word-level bounding boxes.


2. Crops each word box.


3. Segments characters inside using contours.


4. Draws parallelogram-like character boxes.


5. Computes centroids and fits both a curve and a straight line.


6. Compares fit error to decide whether text is curved or straight.


7. Draws a curve (red) or a straight line (blue) accordingly.



Let me know if you'd like visualization or real-time plotting.

