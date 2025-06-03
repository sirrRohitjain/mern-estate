""import cv2 import numpy as np import easyocr import torch

Initialize EasyOCR reader

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def fit_curve_to_parallelograms(parallelograms, image): if len(parallelograms) < 3: return  # Need at least 3 points for a quadratic curve

centroids = []
for box in parallelograms:
    M = cv2.moments(box)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))

if len(centroids) < 3:
    return

centroids = sorted(centroids, key=lambda x: x[0])
x = np.array([pt[0] for pt in centroids])
y = np.array([pt[1] for pt in centroids])

coeffs = np.polyfit(x, y, 2)
poly = np.poly1d(coeffs)
x_curve = np.linspace(min(x), max(x), 100)
y_curve = poly(x_curve)

curve_points = np.array([[int(xi), int(yi)] for xi, yi in zip(x_curve, y_curve)], dtype=np.int32)
cv2.polylines(image, [curve_points.reshape(-1, 1, 2)], False, (0, 0, 255), 2)

def process_image_with_mser_and_ocr(image_path, output_path): image = cv2.imread(image_path) if image is None: print(f"Error: Cannot read image from {image_path}") return

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
results = reader.readtext(image)

mser = cv2.MSER_create()
mser.setMinArea(30)
mser.setMaxArea(5000)

for (bbox, text, conf) in results:
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    x_min = int(min(pt[0] for pt in bbox))
    y_min = int(min(pt[1] for pt in bbox))
    x_max = int(max(pt[0] for pt in bbox))
    y_max = int(max(pt[1] for pt in bbox))

    roi = gray[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        continue

    regions, _ = mser.detectRegions(roi)
    parallelograms = []

    for region in regions:
        if len(region) < 5:
            continue
        rect = cv2.minAreaRect(region)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        box[:, 0] += x_min
        box[:, 1] += y_min
        parallelograms.append(box)
        cv2.polylines(image, [box], isClosed=True, color=(255, 0, 0), thickness=1)

    fit_curve_to_parallelograms(parallelograms, image)

cv2.imwrite(output_path, image)
print(f"Saved result to {output_path}")

if name == "main": input_image = "your_input_image.jpg" output_image = "output_with_curves.jpg" process_image_with_mser_and_ocr(input_image, output_image)

