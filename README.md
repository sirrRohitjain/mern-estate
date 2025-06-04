import cv2
import numpy as np
import easyocr

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)

def segment_characters_in_word(image, word_bbox):
    # Extract word patch using bounding box
    pts = np.array(word_bbox).astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    word_img = image[y:y+h, x:x+w]

    # Convert to grayscale and binarize
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours which should correspond to characters
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_boxes = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if cw > 5 and ch > 5:  # Filter out small noise
            char_boxes.append([x+cx, y+cy, x+cx+cw, y+cy+ch])

    return char_boxes

def process_image(image_path):
    image = cv2.imread(image_path)
    results = reader.readtext(image)

    for (bbox, text, conf) in results:
        pts = np.array(bbox).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Segment characters inside this word bbox
        char_boxes = segment_characters_in_word(image, bbox)
        for (x1, y1, x2, y2) in char_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imwrite("output_character_segmented.png", image)
    print("Saved to output_character_segmented.png")

# Example run
process_image("your_image_path.jpg")