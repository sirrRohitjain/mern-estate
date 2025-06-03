import cv2
import numpy as np
import easyocr

def detect_text_and_curves(image_path, output_path="output.png", curve_degree=2):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    img_draw = image.copy()

    # Initialize OCR reader
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image)

    # Draw OCR bounding boxes and process each word-box with MSER
    mser = cv2.MSER_create(_delta=5, _min_area=30, _max_area=5000)

    for (bbox, text, conf) in results:
        # Draw OCR word bounding box (in green)
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(img_draw, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Define bounding rectangle for MSER
        x_min = int(min(p[0] for p in bbox))
        y_min = int(min(p[1] for p in bbox))
        x_max = int(max(p[0] for p in bbox))
        y_max = int(max(p[1] for p in bbox))
        roi = image[y_min:y_max, x_min:x_max]

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        regions, _ = mser.detectRegions(gray_roi)

        centroids = []

        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = box.astype(np.intp)

            # Shift box to image coordinates
            box[:, 0] += x_min
            box[:, 1] += y_min

            # Draw parallelogram box (in blue)
            cv2.polylines(img_draw, [box], isClosed=True, color=(255, 0, 0), thickness=1)

            # Compute centroid
            M = cv2.moments(box)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        # Fit curve only if enough points
        if len(centroids) >= curve_degree + 1:
            x_coords = np.array([pt[0] for pt in centroids])
            y_coords = np.array([pt[1] for pt in centroids])

            try:
                poly = np.poly1d(np.polyfit(x_coords, y_coords, curve_degree))
                x_range = np.linspace(min(x_coords), max(x_coords), 100)
                y_curve = poly(x_range)
                curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_range, y_curve)], np.int32)
                cv2.polylines(img_draw, [curve_pts.reshape(-1, 1, 2)], isClosed=False, color=(0, 0, 255), thickness=2)
            except np.linalg.LinAlgError:
                pass

    cv2.imwrite(output_path, img_draw)
    print(f"Output saved to {output_path}")