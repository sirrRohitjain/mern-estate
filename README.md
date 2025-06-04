import cv2
import numpy as np
import easyocr

# Load image
image = cv2.imread("your_image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(image_rgb)

# Make a copy for drawing
output = image.copy()

for (bbox, text, conf) in results:
    bbox = np.array(bbox).astype(int)

    # Draw OCR word-level bounding box (in blue)
    cv2.polylines(output, [bbox], isClosed=True, color=(255, 0, 0), thickness=2)

    # Crop the word region
    x_min = np.min(bbox[:, 0])
    x_max = np.max(bbox[:, 0])
    y_min = np.min(bbox[:, 1])
    y_max = np.max(bbox[:, 1])
    word_crop = image[y_min:y_max, x_min:x_max]

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(word_crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected component segmentation
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # Filter out noise
        if area > 30 and w > 2 and h > 5:
            # Draw green boxes for character-level segmentation (adjust to original coordinates)
            cv2.rectangle(output, (x + x_min, y + y_min), (x + x_min + w, y + y_min + h), (0, 255, 0), 1)
            cv2.circle(output, (int(cx + x_min), int(cy + y_min)), 1, (0, 0, 255), -1)

# Save final result
cv2.imwrite("ocr_connected_components_output.jpg", output)