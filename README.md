import cv2
import numpy as np

def is_likely_text_region(contour, bbox, min_solidity=0.4, min_extent=0.3):
    x, y, w, h = bbox
    area = cv2.contourArea(contour)
    if area == 0 or w == 0 or h == 0:
        return False
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    solidity = area / hull_area
    extent = area / (w * h)
    aspect_ratio = w / float(h)
    
    return (
        0.1 < aspect_ratio < 10 and
        solidity > min_solidity and
        extent > min_extent and
        w > 10 and h > 10
    )

def detect_text_regions(image_path, output_path="output_text_boxes.png"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # MSER detector
    mser = cv2.MSER_create(_min_area=100, _max_area=100000)
    regions, bboxes = mser.detectRegions(gray)

    mask = np.zeros_like(gray, dtype=np.uint8)

    # Filter out non-text regions
    for i, contour in enumerate(regions):
        bbox = bboxes[i]
        if is_likely_text_region(contour, bbox):
            cv2.drawContours(mask, [contour], -1, 255, -1)

    # Morphological operations to group close letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    # Find contours of text blobs
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw final bounding boxes
    result = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 15:  # basic size threshold to remove junk
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, result)
    print(f"Saved result with text bounding boxes to: {output_path}")

# --- Run it ---
if __name__ == "__main__":
    detect_text_regions("testt.png", "text_only_boxes.png")