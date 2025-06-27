import cv2
import numpy as np
import os

def preprocess(roi_gray, save_debug=False, debug_name="roi_gray.png"):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi_gray)
    roi_blur = cv2.GaussianBlur(roi_eq, (3, 3), 0)
    if save_debug:
        cv2.imwrite(debug_name, roi_blur)
    return roi_blur

def iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0

def group_bboxes(bboxes, max_x_dist=50, max_y_dist=15, max_height_diff=0.5):
    groups = []
    used = [False] * len(bboxes)

    for i in range(len(bboxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = bboxes[i]
        group = [bboxes[i]]
        used[i] = True
        for j in range(i + 1, len(bboxes)):
            if used[j]:
                continue
            x2, y2, w2, h2 = bboxes[j]
            if abs(y1 - y2) < max_y_dist and abs(h1 - h2) / max(h1, h2) < max_height_diff:
                if abs((x2 + w2 / 2) - (x1 + w1 / 2)) < max_x_dist:
                    group.append(bboxes[j])
                    used[j] = True
        if len(group) >= 3:
            groups.append(group)
    return groups

def fit_and_draw(group, roi_color, color=(0, 0, 255), degree_options=(1, 2, 3)):
    centroids = [(x + w // 2, y + h // 2) for x, y, w, h in group]
    centroids.sort(key=lambda p: p[0])
    cx = np.array([pt[0] for pt in centroids])
    cy = np.array([pt[1] for pt in centroids])

    best_pts = None
    best_score = -1
    for deg in degree_options:
        if len(cx) <= deg:
            continue
        try:
            coeffs = np.polyfit(cx, cy, deg)
            poly = np.poly1d(coeffs)
            x_vals = np.linspace(min(cx), max(cx), 200)
            y_vals = poly(x_vals)
            covered = sum(np.hypot(x - px, y - py) < 15 for px, py in centroids for x, y in zip(x_vals, y_vals))
            if covered > best_score:
                best_score = covered
                best_pts = np.array([[int(x), int(y)] for x, y in zip(x_vals, y_vals)])
        except:
            continue
    if best_pts is not None:
        cv2.polylines(roi_color, [best_pts.reshape(-1, 1, 2)], False, color, 2)

def detect_text(image_path, roi_bbox, output_path="result.png"):
    img = cv2.imread(image_path)
    x, y, w, h = roi_bbox
    roi = img[y:y+h, x:x+w].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    processed = preprocess(gray, save_debug=True, debug_name="preprocessed_roi.png")

    mser = cv2.MSER_create()
    regions, bboxes = mser.detectRegions(processed)

    char_boxes = []
    for bx in bboxes:
        x0, y0, bw, bh = bx
        if bw < 5 or bh < 5 or bw > 300 or bh > 300:
            continue
        ar = bw / float(bh)
        if ar < 0.2 or ar > 2.5:
            continue
        char_boxes.append((x0, y0, bw, bh))
        cv2.rectangle(roi, (x0, y0), (x0 + bw, y0 + bh), (0, 255, 0), 1)

    groups = group_bboxes(char_boxes)
    for group in groups:
        fit_and_draw(group, roi, degree_options=(1, 2, 3))

    img[y:y+h, x:x+w] = roi
    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_file = "./image_testing/image079.jpg"
    roi_box = (930, 857, 347, 124)
    detect_text(image_file, roi_box, output_path="./image_results/image079_output.jpg")