import cv2
import numpy as np
import easyocr
import torch

def extract_characters_with_mser(gray_crop):
    mser = cv2.MSER_create()
    mser.setMinArea(30)
    mser.setMaxArea(5000)
    mser.setDelta(5)
    mser.setMaxVariation(0.25)
    mser.setMinDiversity(0.2)

    regions, _ = mser.detectRegions(gray_crop)
    char_info = []

    for region in regions:
        if len(region) < 5:
            continue
        rect = cv2.minAreaRect(region)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        centroid = np.mean(box, axis=0)
        char_info.append({'box': box, 'centroid': centroid})
    return char_info

def fit_and_draw_curve(image, centroids, color=(0, 0, 255), degree=2):
    if len(centroids) < degree + 1:
        return
    centroids = sorted(centroids, key=lambda p: p[0])
    x_coords = np.array([p[0] for p in centroids])
    y_coords = np.array([p[1] for p in centroids])

    try:
        coeffs = np.polyfit(x_coords, y_coords, degree)
        poly_func = np.poly1d(coeffs)
        x_vals = np.linspace(x_coords.min(), x_coords.max(), 100)
        y_vals = poly_func(x_vals)
        curve_points = np.array([[int(x), int(y)] for x, y in zip(x_vals, y_vals)], np.int32)
        cv2.polylines(image, [curve_points.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=2)
    except np.RankWarning:
        pass

def process_image(image_path, output_path="output.png", curve_degree=2):
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    img = cv2.imread(image_path)
    img_output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(img)

    for bbox, text, conf in results:
        bbox_np = np.array(bbox, dtype=np.int32)
        cv2.polylines(img_output, [bbox_np.reshape(-1, 1, 2)], True, (0, 255, 0), 2)

        x_min = np.min(bbox_np[:, 0])
        y_min = np.min(bbox_np[:, 1])
        x_max = np.max(bbox_np[:, 0])
        y_max = np.max(bbox_np[:, 1])

        word_crop = gray[y_min:y_max, x_min:x_max]
        chars = extract_characters_with_mser(word_crop)

        char_centroids = []
        for char in chars:
            # Shift box coordinates to original image space
            shifted_box = char['box'] + np.array([x_min, y_min])
            shifted_centroid = char['centroid'] + np.array([x_min, y_min])
            char_centroids.append(shifted_centroid)
            cv2.polylines(img_output, [shifted_box.reshape(-1, 1, 2)], True, (255, 0, 0), 1)

        if len(char_centroids) >= curve_degree + 1:
            fit_and_draw_curve(img_output, char_centroids, degree=curve_degree)

    cv2.imwrite(output_path, img_output)
    print(f"Saved output to: {output_path}")

# Example usage
if __name__ == "__main__":
    process_image("your_image.jpg", output_path="curved_text_output.png", curve_degree=2)