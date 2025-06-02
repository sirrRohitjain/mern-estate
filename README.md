import cv2
import numpy as np

def detect_text_blocks(gray, min_area=1000, max_area=100000):
    """
    Detect larger text blocks using MSER or contours.
    """
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    regions, bboxes = mser.detectRegions(gray)
    
    filtered_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        if w < 20 or h < 20:
            continue
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 10:
            filtered_bboxes.append((x, y, w, h))
    return filtered_bboxes

def detect_characters_in_block(gray_crop, min_area=30, max_area=3000):
    """
    Detect character candidates inside a cropped block using MSER.
    """
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    regions, bboxes = mser.detectRegions(gray_crop)
    
    char_candidates = []
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        if w < 5 or h < 5 or w > 200 or h > 200:
            continue
        aspect_ratio = w / float(h)
        if not (0.1 < aspect_ratio < 5):
            continue
        if len(regions[i]) < 5:
            continue
        contour = regions[i]
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        extent = float(area) / (w * h)
        if not (solidity > 0.3 and extent > 0.15):
            continue
        centroid = (x + w // 2, y + h // 2)
        char_candidates.append({"centroid": centroid, "w": w, "h": h, "bbox": (x, y, w, h)})
    return char_candidates

def group_and_fit_curve(chars, curve_degree=2, min_chars=4):
    """
    Group characters left-to-right and fit curve through their centroids.
    """
    chars.sort(key=lambda c: c['centroid'][0])
    if len(chars) < min_chars:
        return None

    x_coords = np.array([c['centroid'][0] for c in chars])
    y_coords = np.array([c['centroid'][1] for c in chars])

    if len(x_coords) < curve_degree + 1:
        return np.array([(int(x), int(y)) for x, y in zip(x_coords, y_coords)], np.int32)

    try:
        poly = np.poly1d(np.polyfit(x_coords, y_coords, curve_degree))
        x_fit = np.linspace(x_coords.min(), x_coords.max(), 100)
        y_fit = poly(x_fit)
        curve = np.array([[int(x), int(y)] for x, y in zip(x_fit, y_fit)], np.int32)
        return curve
    except:
        return np.array([(int(x), int(y)) for x, y in zip(x_coords, y_coords)], np.int32)

def detect_and_draw_curved_text(image_path, output_path="output_text_curves.png"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect large text blocks
    text_blocks = detect_text_blocks(gray)

    for (x, y, w, h) in text_blocks:
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 255, 100), 1)

        # Step 2: Crop block and detect characters
        cropped_gray = gray[y:y+h, x:x+w]
        char_candidates = detect_characters_in_block(cropped_gray)

        for char in char_candidates:
            cx, cy = char['centroid']
            char['centroid'] = (cx + x, cy + y)  # Translate to original coords
            bx, by, bw, bh = char['bbox']
            char['bbox'] = (bx + x, by + y, bw, bh)
            cv2.rectangle(img, (bx + x, by + y), (bx + x + bw, by + y + bh), (255, 200, 0), 1)

        # Step 3: Group and fit curve
        curve = group_and_fit_curve(char_candidates)
        if curve is not None:
            cv2.polylines(img, [curve.reshape(-1, 1, 2)], False, (0, 0, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"Saved output with text curves to: {output_path}")

# --- Run ---
if __name__ == "__main__":
    detect_and_draw_curved_text("testt.png", "output_curved_lines.png")