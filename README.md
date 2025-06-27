import cv2
import numpy as np
import os

# Step 1: Color Reduction (K-means)
def color_reduction_kmeans(image, k=4):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    reduced = centers[labels.flatten()].reshape((image.shape))
    return reduced

# Step 2: Gradient-Based Saliency Approximation
def compute_gradient_saliency(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    grad_mag = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
    _, mask = cv2.threshold(grad_mag, 80, 255, cv2.THRESH_BINARY)
    return grad_mag, mask

# Step 3: Preprocessing ROI with color reduction and saliency
def extract_text_regions_color_saliency(image_path, roi, debug_prefix="debug"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")

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

    return text_region, saliency_mask, roi_color, gray

# Step 4: MSER + Filtering
def get_mser_characters(masked_image, original_roi, processed_roi=None, min_area=30, max_area=10000, delta=5, save_debug=False, debug_prefix="debug"):
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setDelta(delta)

    regions, bboxes = mser.detectRegions(masked_image)
    char_centroids = []

    for box in bboxes:
        x, y, w, h = box
        if w < 2 or h < 2 or w > 800 or h > 800:
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.05 or aspect_ratio > 10:
            continue

        cx = x + w // 2
        cy = y + h // 2
        char_centroids.append((float(cx), float(cy)))

        if save_debug:
            cv2.rectangle(original_roi, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.circle(original_roi, (int(cx), int(cy)), 2, (0, 0, 255), -1)
            if processed_roi is not None:
                cv2.rectangle(processed_roi, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.circle(processed_roi, (int(cx), int(cy)), 2, (0, 0, 255), -1)

    if save_debug:
        cv2.imwrite(f"{debug_prefix}_mser_detected.jpg", original_roi)
        if processed_roi is not None:
            cv2.imwrite(f"{debug_prefix}_processed_roi_with_mser.jpg", processed_roi)

    return char_centroids, original_roi

# Step 4B: Add contour-based candidates
def add_contour_boxes(masked_image, centroids, debug_img):
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10 or w > 800 or h > 800:
            continue
        cx = x + w // 2
        cy = y + h // 2
        if not any(np.hypot(cx - ecx, cy - ecy) < 10 for ecx, ecy in centroids):
            centroids.append((cx, cy))
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 200, 255), 1)
            cv2.circle(debug_img, (int(cx), int(cy)), 2, (200, 0, 255), -1)
    return centroids, debug_img

# Step 4C: Fallback if not enough centroids
def fallback_otsu_contour(gray_roi):
    _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fallback_centroids = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 100 or w > 800 or h > 800:
            continue
        cx = x + w // 2
        cy = y + h // 2
        fallback_centroids.append((cx, cy))
    return fallback_centroids

# Step 5: Fit and draw curve through centroids
def fit_and_draw_curve(centroids, roi_image, degree_options=(1, 2, 3), max_dist=10):
    if len(centroids) < 3:
        print("Not enough centroids for curve fitting.")
        return roi_image

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

# Step 6: Full processing pipeline
def process_single_text_roi(image_path, roi, debug_prefix="debug"):
    masked_roi, saliency_mask, original_roi, gray = extract_text_regions_color_saliency(image_path, roi, debug_prefix=debug_prefix)

    # ✅ Save the processed ROI used as input to MSER
    cv2.imwrite(f"{debug_prefix}_processed_roi.jpg", masked_roi)

    centroids, roi_with_boxes = get_mser_characters(
        masked_roi,
        original_roi.copy(),
        processed_roi=masked_roi.copy(),
        save_debug=True,
        debug_prefix=debug_prefix
    )
    centroids, roi_with_boxes = add_contour_boxes(masked_roi, centroids, roi_with_boxes)

    if len(centroids) < 3:
        print("Too few MSER + contour centroids, using fallback.")
        centroids = fallback_otsu_contour(gray)

    final_result = fit_and_draw_curve(centroids, roi_with_boxes)
    cv2.imwrite(f"{debug_prefix}_final_result.jpg", final_result)
    print(f"Saved final result to {debug_prefix}_final_result.jpg")

# Example usage
if __name__ == "__main__":
    image_file = "./image_testing/image079.jpg"
    roi_box = (930, 857, 347, 124)  # (x, y, w, h)
    process_single_text_roi(image_file, roi_box, debug_prefix="./image_results/image079")


✅ Complete code updated to:

Draw MSER boxes and centroids on both original ROI and processed ROI

Save *_processed_roi_with_mser.jpg for visual inspection


You’ll now see:

processed_roi.jpg – input to MSER

processed_roi_with_mser.jpg – same, with green boxes and red dots

final_result.jpg – fitted curve overlaid


Let me know if you want:

Character boxes as parallelograms

Bounding polygon around all characters

Centroid data exported to a file (JSON/CSV)


