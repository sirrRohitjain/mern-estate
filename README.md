import cv2
import easyocr
import numpy as np

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    resized = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized

def draw_boxes(img, boxes, color=(0,255,0), thickness=2, texts=None):
    for i, box in enumerate(boxes):
        pts = np.array(box, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        if texts:
            cv2.putText(img, texts[i], tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    image_path = 'your_image.jpg'  # <- Replace this
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return

    pre_img = preprocess(img)
    pre_img_rgb = cv2.cvtColor(pre_img, cv2.COLOR_GRAY2RGB)

    reader = easyocr.Reader(['en'])

    # Word-level results
    word_results = reader.readtext(pre_img_rgb, detail=2)

    # Character-level results
    char_results = reader.readtext(pre_img_rgb, detail=1, paragraph=False, contrast_ths=0.1, adjust_contrast=0.5, decoder='beamsearch')

    # Create image to draw
    img_draw = pre_img_rgb.copy()

    # Draw word-level boxes (green)
    word_boxes = [res[0] for res in word_results]
    word_texts = [res[1] for res in word_results]
    draw_boxes(img_draw, word_boxes, (0,255,0), thickness=2, texts=word_texts)

    # Draw character-level boxes (blue)
    for word in char_results:
        if hasattr(word, 'char_boxes'):
            for cbox in word.char_boxes:
                pts = np.array(cbox, dtype=np.int32).reshape((-1,1,2))
                cv2.polylines(img_draw, [pts], isClosed=True, color=(255,0,0), thickness=1)

    # Show result
    cv2.imshow("Detected Word (Green) and Character (Blue) Boxes", img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output_with_bounding_boxes.jpg", img_draw)
    print("Saved as output_with_bounding_boxes.jpg")

if __name__ == "__main__":
    main()