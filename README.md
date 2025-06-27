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

def get_saliency_mask(image, saliency_threshold=128):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliency_map) = saliency.computeSaliency(image)
    if not success:
        raise ValueError("Saliency computation failed.")
    
    saliency_map = (saliency_map * 255).astype(np.uint8)
    _, saliency_mask = cv2.threshold(saliency_map, saliency_threshold, 255, cv2.THRESH_BINARY)
    return saliency_map, saliency_mask

def extract_text_regions_color_saliency(image_path, debug_prefix="debug"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")

    # 1. Color Reduction
    reduced = color_reduction_kmeans(img, k=4)
    cv2.imwrite(f"{debug_prefix}_color_reduced.jpg", reduced)

    # 2. Saliency Detection
    saliency_map, saliency_mask = get_saliency_mask(reduced)
    cv2.imwrite(f"{debug_prefix}_saliency_map.jpg", saliency_map)
    cv2.imwrite(f"{debug_prefix}_saliency_mask.jpg", saliency_mask)

    # 3. Apply mask to image to keep only probable text regions
    result = cv2.bitwise_and(img, img, mask=saliency_mask)
    cv2.imwrite(f"{debug_prefix}_text_region_candidates.jpg", result)

    return result, saliency_mask

# Example usage
if __name__ == "__main__":
    image_file = "./image_testing/image079.jpg"
    extract_text_regions_color_saliency(image_file, debug_prefix="./image_results/image079")