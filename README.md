import cv2 import numpy as np import os

Step 1: Color Reduction (K-means)

def color_reduction_kmeans(image, k=4): Z = image.reshape((-1, 3)) Z = np.float32(Z) criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) centers = np.uint8(centers) reduced = centers[labels.flatten()].reshape((image.shape)) return reduced

Step 2: Gradient-Based Saliency Approximation

def compute_gradient_saliency(gray): sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2) grad_mag = (grad_mag / grad_mag.max() * 255).astype(np.uint8) _, mask = cv2.threshold(grad_mag, 80, 255, cv2.THRESH_BINARY) return grad_mag, mask

Step 3: Preprocessing ROI with color reduction and saliency

def extract_text_regions_color_saliency(image_path, roi, debug_prefix="debug"): img = cv2.imread(image_path) if img is None: raise FileNotFoundError("Image not found")

x, y, w, h = roi
roi_color = img[y:y+h, x:x+w].copy()

reduced = color_reduction_kmeans(roi_color, k=4)
cv2.imwrite(f"{debug_prefix}_color_reduced.jpg", reduced)

gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
grad_mag, saliency_mask = compute_gradient_saliency(gray)
cv2.imwrite(f"{debug_prefix}_gradient_saliency.jpg", grad_mag)
cv2.imwrite(f"{debug_prefix}_saliency_mask.jpg", saliency_mask)

text_region = cv2.bitwise_and(roi_color, roi_color, mask=saliency_mask)
cv2.imwrite(f"{debug_prefix}_text_region_candidates.jpg", text_region)

return text_region, saliency_mask, roi_color

Step 4: MSER + Filtering

def get_mser_characters(masked_image, original_roi, min_area=30, max_area=10000, delta=5, save_debug=False, debug_prefix="debug"): mser = cv2.MSER_create() mser.setMinArea(min_area) mser.setMaxArea(max_area) mser.setDelta(delta)

regions, bboxes = mser.detectRegions(masked_image)
char_centroids = []

for box in bboxes:
    x, y, w, h = box
    if w < 5 or h < 5 or w > 300 or h > 300:
        continue
    aspect_ratio = w / float(h)
    if aspect_ratio < 0.2 or aspect_ratio > 2.5:
        continue

    cx = x + w // 2
    cy = y + h // 2
    char_centroids.append((float(cx), float(cy)))

    if save_debug:
        cv2.rectangle(original_roi, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.circle(original_roi, (int(cx), int(cy)), 2, (0, 0, 255), -1)

if save_debug:
    cv2.imwrite(f"{debug_prefix}_mser_detected.jpg", original_roi)

return char_centroids, original_roi

Step 5: Fit and draw curve through centroids

def fit_and_draw_curve(centroids, roi_image, degree_options=(1, 2, 3), max_dist=10): if len(centroids) < 3: print("Not enough centroids for curve fitting.") return roi_image

centroids = sorted(centroids, key=lambda pt: pt[0])
cx = np.array([pt[0] for pt in centroids])
cy = np.array([pt[1] for pt in centroids])

best_curve_pts = None
best_score = -1
best_deg = None

for deg in degree_options:
    if len(cx) <= deg:
        continue
    try:
        coeffs = np.polyfit(cx, cy, deg)
        poly = np.poly1d(coeffs)
        x_curve = np.linspace(min(cx), max(cx), 200)
        y_curve = poly(x_curve)

        score = sum(np.hypot(x - cx_i, y - cy_i) < max_dist for cx_i, cy_i in zip(cx, cy) for x, y in zip(x_curve, y_curve))
        if score > best_score:
            best_score = score
            best_curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)])
            best_deg = deg
    except:
        continue

if best_curve_pts is not None:
    cv2.polylines(roi_image, [best_curve_pts.reshape(-1, 1, 2)], False, (255, 0, 0), 2)
    print(f"Fitted curve: degree={best_deg}, coverage={best_score}")
else:
    pt1 = (int(cx[0]), int(cy[0]))
    pt2 = (int(cx[-1]), int(cy[-1]))
    cv2.line(roi_image, pt1, pt2, (255, 0, 255), 2)
    print("Fallback to straight line.")

return roi_image

Step 6: Full processing pipeline

def process_single_text_roi(image_path, roi, debug_prefix="debug"): masked_roi, saliency_mask, original_roi = extract_text_regions_color_saliency(image_path, roi, debug_prefix=debug_prefix) centroids, roi_with_boxes = get_mser_characters(masked_roi, original_roi.copy(), save_debug=True, debug_prefix=debug_prefix) final_result = fit_and_draw_curve(centroids, roi_with_boxes) cv2.imwrite(f"{debug_prefix}_final_result.jpg", final_result) print(f"Saved final result to {debug_prefix}_final_result.jpg")

Example usage

if name == "main": image_file = "./image_testing/image079.jpg" roi_box = (930, 857, 347, 124)  # (x, y, w, h) process_single_text_roi(image_file, roi_box, debug_prefix="./image_results/image079")

