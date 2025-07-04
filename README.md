import cv2
import numpy as np
import os


def kmeans(image, k=4):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    reduced = centers[labels.flatten()].reshape((image.shape))
    cv2.imshow("image",reduced)
    cv2.waitKey(0)
    return reduced

def saliency(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    grad_mag = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
    _, mask = cv2.threshold(grad_mag, 80, 255, cv2.THRESH_BINARY)
    return grad_mag, mask


def extract_text_regions_color_saliency(image_path, roi, debug_prefix="debug"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")

    x, y, w, h = roi
    roi_color = img[y:y+h, x:x+w].copy()

    reduced = kmeans(roi_color, k=4)
    cv2.imwrite(f"{debug_prefix}_color_reduced.jpg", reduced)

    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    grad_mag, saliency_mask = saliency(gray)
    cv2.imwrite(f"{debug_prefix}_gradient_saliency.jpg", grad_mag)
    cv2.imwrite(f"{debug_prefix}_saliency_mask.jpg", saliency_mask)

    text_region = cv2.bitwise_and(roi_color, roi_color, mask=saliency_mask)
    cv2.imwrite(f"{debug_prefix}_text_region_candidates.jpg", text_region)

    return text_region, saliency_mask, roi_color, gray

def mser(masked_image, original_roi, processed_roi=None, min_area=30, max_area=10000, delta=5, save_debug=False, debug_prefix="debug"):
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

def otsu_contour(gray_roi):
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

def detect(image_path, roi, debug_prefix="debug"):
    masked_roi, saliency_mask, original_roi, gray = extract_text_regions_color_saliency(image_path, roi, debug_prefix=debug_prefix)

    
    cv2.imwrite(f"{debug_prefix}_processed_roi.jpg", masked_roi)

    centroids, roi_with_boxes = mser(
        masked_roi,
        original_roi.copy(),
        processed_roi=masked_roi.copy(),
        save_debug=True,
        debug_prefix=debug_prefix
    )
    centroids, roi_with_boxes = add_contour_boxes(masked_roi, centroids, roi_with_boxes)

    if len(centroids) < 3:
        print("Too few MSER + contour centroids, using fallback.")
        centroids = otsu_contour(gray)

    final_result = fit_and_draw_curve(centroids, roi_with_boxes)
    cv2.imwrite(f"{debug_prefix}_final_result.jpg", final_result)
    print(f"Saved final result to {debug_prefix}_final_result.jpg")

if __name__ == "__main__":
    image_file = "./image_testing/image053.jpg"
    roi_box = (440,172,598,265)   # (x, y, w, h)
    detect(image_file, roi_box, debug_prefix="./image_results/image053")





    #this method will work only if we just give curve text as an image i.e image should have only text and for regular characters this code 
##this code can be improved a bit by just handling the centroids
import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img, binary

def baselinecurve(binary_image, original_image):

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bottom_points = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:  
            
            bottom = tuple(contour[contour[:, :, 1].argmax()][0])
            bottom_points.append(bottom)
    
    if not bottom_points:
        return original_image
    

    bottom_points = np.array(bottom_points)
    bottom_points = bottom_points[bottom_points[:, 0].argsort()]
    
   
    x = bottom_points[:, 0]
    y = bottom_points[:, 1]
    
   
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)
    
   
    x_curve = np.linspace(x.min(), x.max(), 100)
    y_curve = poly(x_curve)
    curve_points = np.column_stack((x_curve, y_curve)).astype(np.int32)
    
    
    result = original_image.copy()
    cv2.polylines(result, [curve_points], isClosed=False, color=(0, 0, 255), thickness=2)
    
    # Also draw the bottom points for reference
    for point in bottom_points:
        cv2.circle(result, tuple(point), 3, (255, 0, 0), -1)
    
    return result

def main(image_path):
       original, binary = preprocess_image(image_path)
       result_image = baselinecurve(binary, original)
       cv2.imshow('Original Image', original)
       cv2.imshow('Detected Baseline Curve', result_image)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
        
       return result_image
 

if __name__ == "__main__":
    image_path = "curve.png"  
    result_img = main(image_path)
    
    # To save the result
    # cv2.imwrite('detected_curve.jpg', result_img) 


    '''

    def detect_text_topline_curve(binary_image, original_image):
    """Detect the top-line curve of text using contour tops"""
    # Find contours of text components
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Collect TOP points of each contour (approximating top-line)
    top_points = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter small noise
            # Get the TOP-most point of the contour (minimum Y value)
            top = tuple(contour[contour[:, :, 1].argmin()][0])
            top_points.append(top)
    
    if not top_points:
        return original_image
    
    # Convert to numpy array and sort by x-coordinate
    top_points = np.array(top_points)
    top_points = top_points[top_points[:, 0].argsort()]
    
    # Fit a polynomial curve to the top points
    x = top_points[:, 0]
    y = top_points[:, 1]
    
    # Fit a 2nd degree polynomial
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)
    
    # Generate points for the fitted curve
    x_curve = np.linspace(x.min(), x.max(), 100)
    y_curve = poly(x_curve)
    curve_points = np.column_stack((x_curve, y_curve)).astype(np.int32)
    
    # Draw the curve on the original image
    result = original_image.copy()
    cv2.polylines(result, [curve_points], isClosed=False, color=(0, 255, 0), thickness=2)  # Green for top-line
    
    # Draw the top points for reference
    for point in top_points:
        cv2.circle(result, tuple(point), 3, (255, 0, 0), -1)  # Blue dots
    
    return result

    '''
