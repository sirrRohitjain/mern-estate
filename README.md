import cv2
import numpy as np
import easyocr
from sklearn.cluster import DBSCAN

def compute_centroid(box):
    x = [pt[0] for pt in box]
    y = [pt[1] for pt in box]
    return (sum(x) / 4, sum(y) / 4)

def fit_and_draw_curve(image, centroids, color=(0, 0, 255), curve_degree=2):
    centroids.sort(key=lambda pt: pt[0])  # sort left to right
    x_coords = np.array([pt[0] for pt in centroids])
    y_coords = np.array([pt[1] for pt in centroids])

    if len(x_coords) >= curve_degree + 1:
        coeffs = np.polyfit(x_coords, y_coords, curve_degree)
        poly_func = np.poly1d(coeffs)

        x_fit = np.linspace(x_coords.min(), x_coords.max(), 100)
        y_fit = poly_func(x_fit)

        curve_points = np.array([[int(x), int(y)] for x, y in zip(x_fit, y_fit)], np.int32)
        cv2.polylines(image, [curve_points.reshape(-1, 1, 2)], False, color, thickness=2)

def draw_boxes_and_text(image, results):
    for (box, text, conf) in results:
        if conf < 0.5:
            continue
        box = np.int32(box)
        cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.putText(image, text, (box[0][0], box[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def group_by_lines(centroids, eps=50, min_samples=1):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(np.array(centroids))
    line_groups = {}
    for i, label in enumerate(labels):
        if label not in line_groups:
            line_groups[label] = []
        line_groups[label].append(centroids[i])
    return line_groups.values()

def detect_and_fit_text_curves(image_path, output_path, langs=['en'], curve_degree=2):
    image = cv2.imread(image_path)
    reader = easyocr.Reader(langs, gpu=False)
    results = reader.readtext(image_path, detail=1)

    draw_boxes_and_text(image, results)

    centroids = []
    for (box, text, conf) in results:
        if conf >= 0.5:
            centroids.append(compute_centroid(box))

    line_groups = group_by_lines(centroids, eps=60, min_samples=2)

    for group in line_groups:
        if len(group) >= curve_degree + 1:
            fit_and_draw_curve(image, group, curve_degree=curve_degree)

    cv2.imwrite(output_path, image)
    print(f"✅ Output saved to: {output_path}")

# === Main Function Call ===
if __name__ == "__main__":
    detect_and_fit_text_curves("your_image.png", "output_curve_text.png", langs=['en'], curve_degree=2)