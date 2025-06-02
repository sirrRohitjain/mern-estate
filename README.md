import cv2
import numpy as np

def detect_text_in_manual_region(image_path, region, output_path="output.png",
                                 min_area=30, max_area=5000, delta=5,
                                 max_char_dist_x_ratio=1.5, max_char_dist_y_ratio=0.6,
                                 max_height_diff_ratio=0.3, curve_degree=2, min_chars_in_line=3):
    """
    Detect whether the text in a manually specified region is curved or straight.

    Args:
        image_path (str): Path to input image.
        region (tuple): (x, y, w, h) specifying the region to analyze.
        output_path (str): Path to save output image.
        Other parameters: tuning MSER and grouping.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image.")
        return

    x_roi, y_roi, w_roi, h_roi = region
    roi = img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create()
    mser.setDelta(delta)
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)

    regions, bboxes = mser.detectRegions(gray_roi)

    candidate_chars = []
    for i, box in enumerate(bboxes):
        x, y, w, h = box
        if w < 5 or h < 5 or w > 500 or h > 500:
            continue

        aspect_ratio = w / float(h)
        if not (0.1 < aspect_ratio < 10):
            continue

        if len(regions[i]) < 5:
            continue

        contour = regions[i]
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = float(area) / hull_area
        extent = float(area) / (w * h)

        if solidity < 0.3 or extent < 0.15:
            continue

        cx = x + w // 2
        cy = y + h // 2
        candidate_chars.append({"centroid": (cx, cy), "w": w, "h": h, "bbox": (x, y, w, h)})

    # Draw MSER boxes (optional)
    output = roi.copy()
    for c in candidate_chars:
        x, y, w, h = c['bbox']
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

    candidate_chars.sort(key=lambda item: item['centroid'][0])

    lines = []
    used = set()
    for i, c1 in enumerate(candidate_chars):
        if i in used:
            continue
        line = [c1]
        used.add(i)

        for j, c2 in enumerate(candidate_chars):
            if j in used:
                continue

            last = line[-1]
            avg_w = (last['w'] + c2['w']) / 2
            avg_h = (last['h'] + c2['h']) / 2

            dx = c2['centroid'][0] - last['centroid'][0]
            dy = abs(c2['centroid'][1] - last['centroid'][1])
            height_diff = abs(last['h'] - c2['h']) / max(last['h'], c2['h'])

            if 0 < dx < avg_w * max_char_dist_x_ratio and \
               dy < avg_h * max_char_dist_y_ratio and \
               height_diff < max_height_diff_ratio:
                line.append(c2)
                used.add(j)

        if len(line) >= min_chars_in_line:
            lines.append(line)

    for line in lines:
        x_coords = np.array([c['centroid'][0] for c in line])
        y_coords = np.array([c['centroid'][1] for c in line])

        if len(x_coords) < curve_degree + 1:
            points = np.array([c['centroid'] for c in line], np.int32)
            cv2.polylines(output, [points.reshape(-1, 1, 2)], False, (255, 0, 0), 2)
            continue

        coeffs = np.polyfit(x_coords, y_coords, curve_degree)
        poly = np.poly1d(coeffs)

        x_fit = np.linspace(x_coords.min(), x_coords.max(), 100)
        y_fit = poly(x_fit)

        curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_fit, y_fit)], np.int32)
        cv2.polylines(output, [curve_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)

    # Replace the ROI in original image
    img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = output
    cv2.imwrite(output_path, img)
    print(f"Result saved to {output_path}")


# Example usage:
if __name__ == "__main__":
    image_file = "your_image.png"
    manual_region = (50, 100, 300, 100)  # x, y, w, h (in pixels)
    detect_text_in_manual_region(image_file, manual_region, output_path="manual_output.png")


Here's the complete Python code that:

Accepts a manually specified rectangular region (x, y, w, h).

Uses MSER to detect characters within that region.

Groups the detected characters into lines.

Fits a curve to each line to determine if it's curved or straight.

Draws the result and saves the image.


You can find this full implementation in the code block above. Let me know if you want to visualize step-by-step detections or label the output with "Curved" or "Straight" text annotations.

