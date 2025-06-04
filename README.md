import cv2
import easyocr
import numpy as np

def preprocess(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)
    # Resize image (scale by 2)
    resized = cv2.resize(cl_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Make sure dtype is uint8
    return resized.astype(np.uint8)

def draw_boxes(img, boxes, color=(0,255,0), thickness=2, label_boxes=None, label_color=(0,255,0), label_scale=0.7):
    for i, box in enumerate(boxes):
        pts = np.array(box, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        if label_boxes and i < len(label_boxes):
            text = label_boxes[i]
            cv2.putText(img, text, tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_color, 2)

def main():
    image_path = 'your_image.jpg'  # <-- Replace with your image path
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print("Error: Image not found or cannot be loaded.")
        return

    img_pre = preprocess(img_orig)

    reader = easyocr.Reader(['en'])

    # Word-level detection
    word_results = reader.readtext(img_pre, detail=2)

    # Character-level detection using get_textbox (hidden API)
    char_results = reader.get_textbox(
        img_pre,
        x_ths=0.2,
        y_ths=0.6,
        width_ths=0.8,
        decoder='beamsearch'
    )

    # Prepare image for drawing (convert grayscale to BGR)
    img_draw = cv2.cvtColor(img_pre, cv2.COLOR_GRAY2BGR)

    # Draw word boxes in GREEN with labels
    word_boxes = [res[0] for res in word_results]
    word_texts = [res[1] for res in word_results]
    draw_boxes(img_draw, word_boxes, color=(0,255,0), thickness=2, label_boxes=word_texts, label_color=(0,255,0), label_scale=0.7)

    # Draw character boxes in BLUE (no labels)
    char_boxes = [res[0] for res in char_results]
    draw_boxes(img_draw, char_boxes, color=(255,0,0), thickness=1)

    # Show image with bounding boxes
    cv2.imshow("Word (green) and Character (blue) Bounding Boxes", img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output image if you want
    cv2.imwrite("output_bounding_boxes.jpg", img_draw)
    print("Output saved as output_bounding_boxes.jpg")

if __name__ == "__main__":
    main()