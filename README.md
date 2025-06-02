import cv2
import numpy as np

def is_likely_text_region(contour, bbox, gray):
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

    roi = gray[y:y+h, x:x+w]
    mean_intensity = cv2.mean(roi)[0] if roi.size > 0 else 0

    return (
        0.2 < aspect_ratio < 5 and
        0.4 < solidity < 1 and
        0.2 < extent < 0.9 and
        w > 10 and h > 10 and
        20 < mean_intensity < 240
    )

def detect_text_regions(image_path, output_path="output_filtered_text.png"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(5000)

    regions, bboxes = mser.detectRegions(gray)

    mask = np.zeros_like(gray, dtype=np.uint8)
    for i, contour in enumerate(regions):
        bbox = bboxes[i]
        if is_likely_text_region(contour, bbox, gray):
            cv2.drawContours(mask, [contour], -1, 255, -1)

    # Merge nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 15:
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, output_img)
    print(f"Saved result to: {output_path}")

# === Run with input file ===
if __name__ == "__main__":
    input_image_path = "testt.png"     # <<<<<<<<<<<<<<  Replace this with your input image
    output_image_path = "filtered_text_output.png"

    detect_text_regions(input_image_path, output_image_path)