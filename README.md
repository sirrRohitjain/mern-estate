import cv2
import numpy as np
import easyocr
from sklearn.cluster import DBSCAN

def preprocess_image_for_ocr(image_path, output_path="preprocessed.jpg"):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scaled = cv2.resize(binary, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(scaled, None, h=30)
    cv2.imwrite(output_path, denoised)
    return output_path

def group_text_by_lines(centroids, eps=50, min_samples=2):
    if len(centroids) == 0:
        return []

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(np.array(centroids))

    grouped = {}
    for label, centroid in zip(labels, centroids):
        if label == -1:
            continue
        grouped.setdefault(label, []).append(centroid)
    
    return list(grouped.values())

def draw_bounding_boxes_and_text(img, results):
    for bbox, text, conf in results:
        if conf < 0.3 or len(text.strip()) == 0:
            continue
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        text_pos = tuple(pts[0][0])
        cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return img

def draw_fitted_curves_on_image(img, text_lines, curve_degree=2):
    for line in text_lines:
        if len(line) < curve_degree + 1:
            continue
        x = np.array([pt[0] for pt in line])
        y = np.array([pt[1] for pt in line])

        try:
            coeffs = np.polyfit(x, y, deg=curve_degree)
            poly_func = np.poly1d(coeffs)
            x_new = np.linspace(x.min(), x.max(), 100)
            y_new = poly_func(x_new)
            points = np.array([[int(xv), int(yv)] for xv, yv in zip(x_new, y_new)], np.int32)
            cv2.polylines(img, [points.reshape((-1, 1, 2))], isClosed=False, color=(0, 0, 255), thickness=2)
        except Exception as e:
            print(f"Curve fitting failed for a line: {e}")
    return img

def detect_and_fit_text_curves(image_path, output_image="output_full.png", langs=['en'], curve_degree=2):
    preprocessed_path = preprocess_image_for_ocr(image_path)
    reader = easyocr.Reader(langs, gpu=False)
    result = reader.readtext(preprocessed_path)

    centroids = []
    for bbox, text, conf in result:
        if len(text.strip()) == 0 or conf < 0.3:
            continue
        pts = np.array(bbox)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        centroids.append((cx, cy))

    text_lines = group_text_by_lines(centroids, eps=60, min_samples=2)
    img = cv2.imread(image_path)
    img = draw_bounding_boxes_and_text(img, result)
    img = draw_fitted_curves_on_image(img, text_lines, curve_degree)

    cv2.imwrite(output_image, img)
    print(f"[✔] Output image saved: {output_image}")

# ---- Run this block directly ----
if __name__ == "__main__":
    input_image = "curve1.jpg"               # Replace with your file
    output_image = "output_curve_full.png"   # Final result
    detect_and_fit_text_curves(input_image, output_image)