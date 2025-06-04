import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing function (grayscale + CLAHE + resize)
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)
    resized = cv2.resize(cl_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized

# Function to get center of a box
def box_center(box):
    return np.mean(box, axis=0)

# Initialize reader
reader = easyocr.Reader(['en'])

# Load and preprocess image
image_path = 'your_image.jpg'  # replace with your image path
img_original = cv2.imread(image_path)
img_preprocessed = preprocess(img_original)

# Get word-level detection
word_results = reader.readtext(img_preprocessed, detail=2)

# Get character-level detection (hidden EasyOCR function)
char_results = reader.get_textbox(img_preprocessed, x_ths=0.2, y_ths=0.6, width_ths=0.8, decoder='beamsearch')

# Copy image for drawing
img_draw = cv2.cvtColor(img_preprocessed, cv2.COLOR_GRAY2BGR)

# Draw word bounding boxes in GREEN
for word_bbox, word_text, word_prob in word_results:
    word_box_np = np.array(word_bbox, dtype=np.int32)
    cv2.polylines(img_draw, [word_box_np.reshape((-1,1,2))], isClosed=True, color=(0,255,0), thickness=2)
    # Put text label
    cv2.putText(img_draw, word_text, tuple(word_box_np[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# Draw character bounding boxes in BLUE
for char_bbox, char_text, char_prob in char_results:
    char_box_np = np.array(char_bbox, dtype=np.int32)
    cv2.polylines(img_draw, [char_box_np.reshape((-1,1,2))], isClosed=True, color=(255,0,0), thickness=1)
    # Optionally, put character text label
    # cv2.putText(img_draw, char_text, tuple(char_box_np[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

# Show image with all bounding boxes
cv2.imshow("Word and Character Bounding Boxes", img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result image if you want
cv2.imwrite("highlighted_bounding_boxes.jpg", img_draw)