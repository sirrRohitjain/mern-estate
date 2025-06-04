import cv2
import numpy as np
import easyocr
from skimage.morphology import skeletonize
from scipy.interpolate import UnivariateSpline
import math

def detect_text(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    image = cv2.imread(image_path)
    results = reader.readtext(image)

    return image, results

def binarize_and_skeletonize(gray_crop):
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
    return skeleton

def extract_skeleton_points(skeleton):
    points = np.column_stack(np.where(skeleton > 0))
    return points[:, ::-1]  # (x, y) order

def is_curved(points, threshold=0.01):
    if len(points) < 5:
        return False  # not enough points to decide

    points = points[np.argsort(points[:, 0])]
    x = points[:, 0]
    y = points[:, 1]

    try:
        spline = UnivariateSpline(x, y, k=2, s=1)
        y_fit = spline(x)
        mse = np.mean((y - y_fit) ** 2)
        return mse > threshold
    except:
        return False

def draw_fit(image, box, is_curved_text):
    pts = np.array(box, dtype=np.int32)
    x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
    x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])
    w, h = x_max - x_min, y_max - y_min

    crop = image[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    skeleton = binarize_and_skeletonize(gray)
    points = extract_skeleton_points(skeleton)

    if len(points) < 5:
        return

    points = points + np.array([x_min, y_min])
    points = points[np.argsort(points[:, 0])]
    x = points[:, 0]
    y = points[:, 1]

    if is_curved(points):
        spline = UnivariateSpline(x, y, k=2, s=1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = spline(x_fit)
        for i in range(len(x_fit) - 1):
            cv2.line(image, (int(x_fit[i]), int(y_fit[i])), (int(x_fit[i + 1]), int(y_fit[i + 1])), (0, 255, 0), 2)
    else:
        p1 = (int(x.min()), int(np.mean(y)))
        p2 = (int(x.max()), int(np.mean(y)))
        cv2.line(image, p1, p2, (255, 0, 0), 2)

    # Also draw the OCR word box
    cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

def process_image(image_path, output_path="output_result.png"):
    image, results = detect_text(image_path)

    for box, text, conf in results:
        draw_fit(image, box, is_curved)

    cv2.imwrite(output_path, image)
    print(f"[✓] Result saved to {output_path}")

# Run on your image
process_image("your_image.jpg")