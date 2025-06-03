import cv2
import numpy as np
import easyocr
import math
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
import os


def get_mser_character_boxes(gray_roi):
    mser = cv2.MSER_create()
    mser.setDelta(5)
    mser.setMinArea(30)
    mser.setMaxArea(5000)

    regions, _ = mser.detectRegions(gray_roi)
    char_boxes = []

    for pts in regions:
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        char_boxes.append(box)

    return char_boxes


def get_centroid(parallelogram):
    M = cv2.moments(parallelogram)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        return None


def fit_and_draw_curve(img, points):
    if len(points) < 3:
        return

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    z = np.polyfit(x, y, 2)
    f = np.poly1d(z)

    x_new = np.linspace(min(x), max(x), num=500)
    y_new = f(x_new)

    for i in range(len(x_new) - 1):
        pt1 = (int(x_new[i]), int(y_new[i]))
        pt2 = (int(x_new[i + 1]), int(y_new[i + 1]))
        cv2.line(img, pt1, pt2, (0, 255, 255), 2)


def process_image(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb)

    for (bbox, text, conf) in results:
        pts = np.array(bbox).astype(int)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        x_min = np.min(pts[:, 0])
        x_max = np.max(pts[:, 0])
        y_min = np.min(pts[:, 1])
        y_max = np.max(pts[:, 1])

        roi = img[y_min:y_max, x_min:x_max]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        char_boxes = get_mser_character_boxes(gray_roi)
        centroids = []

        for box in char_boxes:
            centroid = get_centroid(box)
            if centroid:
                c = (centroid[0] + x_min, centroid[1] + y_min)
                centroids.append(c)
                cv2.drawContours(img, [box + np.array([x_min, y_min])], 0, (255, 0, 0), 1)

        if len(centroids) >= 3:
            fit_and_draw_curve(img, centroids)

    output_path = os.path.splitext(image_path)[0] + '_output.png'
    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    image_path = "curve1.jpg"  # Change this to your image file
    process_image(image_path)


The complete and corrected code is now in the canvas. It:

1. Uses EasyOCR to detect word-level bounding boxes.


2. For each bounding box, extracts the ROI and applies MSER to detect character-like regions.


3. Converts each MSER region into a parallelogram (via cv2.minAreaRect).


4. Finds centroids of these character boxes.


5. Fits a quadratic curve through the centroids and draws it.


6. Saves the final image.



Let me know if you want enhancements like filtering noise, tuning curve fitting, or clustering text lines.

