import cv2
import numpy as np

def color_reduction_kmeans(image, k=4):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    reduced = centers[labels.flatten()].reshape((image.shape))
    return reduced

def compute_gradient_saliency(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    grad_mag = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
    _, mask = cv2.threshold(grad_mag, 80, 255, cv2.THRESH_BINARY)
    return grad_mag, mask

def extract_text_regions_fallback(image_path, debug_prefix="debug"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")

    # 1. Color Reduction
    reduced = color_reduction_kmeans(img, k=4)
    cv2.imwrite(f"{debug_prefix}_color_reduced.jpg", reduced)

    # 2. Grayscale & gradient-based saliency approximation
    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    grad_mag, saliency_mask = compute_gradient_saliency(gray)
    cv2.imwrite(f"{debug_prefix}_gradient_saliency.jpg", grad_mag)
    cv2.imwrite(f"{debug_prefix}_saliency_mask.jpg", saliency_mask)

    # 3. Apply mask to keep text-like areas
    result = cv2.bitwise_and(img, img, mask=saliency_mask)
    cv2.imwrite(f"{debug_prefix}_text_region_candidates.jpg", result)

    return result, saliency_mask

# Example usage
if __name__ == "__main__":
    image_file = "./image_testing/image079.jpg"
    extract_text_regions_fallback(image_file, debug_prefix="./image_results/image079")