import cv2
import numpy as np
import easyocr
import torch

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def detect_mser_chars(cropped_img, min_area=10, max_area=1000):
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    
    regions, bboxes = mser.detectRegions(gray)

    filtered = []
    for (x, y, w, h) in bboxes:
        if w < 5 or h < 5:
            continue
        aspect = w / float(h)
        if 0.1 < aspect < 10:
            cx = x + w // 2
            cy = y + h // 2
            filtered.append(((cx, cy), (x, y, w, h)))
    
    print(f"MSER found {len(filtered)} character boxes")
    return filtered

def draw_curve(img, centroids, degree=2, color=(0, 0, 255), thickness=2):
    if len(centroids) < degree + 1:
        print("Not enough centroids to fit curve.")
        return

    centroids = sorted(centroids, key=lambda p: p[0])
    x_coords = np.array([p[0] for p in centroids])
    y_coords = np.array([p[1] for p in centroids])

    try:
        coeffs = np.polyfit(x_coords, y_coords, degree)
        poly = np.poly1d(coeffs)
        x_vals = np.linspace(x_coords.min(), x_coords.max(), 100)
        y_vals = poly(x_vals)
        curve_points = np.array([[int(x), int(y)] for x, y in zip(x_vals, y_vals)], np.int32)
        cv2.polylines(img, [curve_points.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness)
        print("Curve drawn.")
    except Exception as e:
        print(f"Curve fitting failed: {e}")

def process_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error reading image.")
        return

    results = reader.readtext(img)

    if not results:
        print("No text detected by EasyOCR.")
        return

    for bbox, text, conf in results:
        pts = np.array(bbox).astype(int)
        x_min = np.min(pts[:, 0])
        y_min = np.min(pts[:, 1])
        x_max = np.max(pts[:, 0])
        y_max = np.max(pts[:, 1])
        
        word_crop = img[y_min:y_max, x_min:x_max]
        if word_crop.size == 0:
            print("Empty crop, skipping.")
            continue

        mser_chars = detect_mser_chars(word_crop)
        char_centroids = []

        for (cx, cy), (x, y, w, h) in mser_chars:
            cv2.rectangle(img, (x_min + x, y_min + y), (x_min + x + w, y_min + y + h), (0, 255, 0), 1)
            char_centroids.append((x_min + cx, y_min + cy))

        draw_curve(img, char_centroids)

    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")