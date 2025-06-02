import cv2
import numpy as np

def detect_and_draw_curved_text_area(image_path, output_image_path="output_curved_text_areas.png",
                                     min_area=30, max_area=5000, max_variation=0.25,
                                     min_diversity=0.2, delta=5,
                                     max_char_dist_x_ratio=1.5, max_char_dist_y_ratio=0.8,
                                     max_height_diff_ratio=0.3, curve_degree=2, min_chars_in_line=3):

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setMaxVariation(max_variation)
    mser.setMinDiversity(min_diversity)
    mser.setDelta(delta)

    regions, bboxes = mser.detectRegions(gray)

    candidate_chars = []
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        if w < 5 or h < 5 or w > 500 or h > 500:
            continue

        aspect_ratio = w / float(h)
        if not (0.05 < aspect_ratio < 15):
            continue

        if len(regions[i]) < 5:
            continue

        contour = regions[i]
        area = cv2.contourArea(contour)
        if area == 0:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = float(area) / hull_area
        extent = float(area) / (w * h)

        if not (solidity > 0.3 and extent > 0.15):
            continue

        centroid = (x + w // 2, y + h // 2)
        candidate_chars.append({"centroid": centroid, "w": w, "h": h, "bbox": bbox})

    img_output = img.copy()
    for char_data in candidate_chars:
        x, y, w, h = char_data['bbox']
        cv2.rectangle(img_output, (x, y), (x + w, y + h), (100, 255, 100), 1)

    candidate_chars.sort(key=lambda item: item["centroid"][0])
    text_lines = []
    used_indices = set()

    for i, char1 in enumerate(candidate_chars):
        if i in used_indices:
            continue

        current_line_indices = [i]
        current_line_chars = [char1]
        used_indices.add(i)

        for j, char2 in enumerate(candidate_chars):
            if j in used_indices:
                continue

            last_char_in_line = current_line_chars[-1]
            avg_width = (last_char_in_line['w'] + char2['w']) / 2
            dist_x = char2['centroid'][0] - last_char_in_line['centroid'][0]
            if not (0 < dist_x < avg_width * max_char_dist_x_ratio):
                continue

            avg_height = (last_char_in_line['h'] + char2['h']) / 2
            dist_y_centroid = abs(last_char_in_line['centroid'][1] - char2['centroid'][1])
            if dist_y_centroid > avg_height * max_char_dist_y_ratio:
                continue

            height_diff_ratio = abs(last_char_in_line['h'] - char2['h']) / max(last_char_in_line['h'], char2['h'])
            if height_diff_ratio > max_height_diff_ratio:
                continue

            current_line_indices.append(j)
            current_line_chars.append(char2)
            used_indices.add(j)
            current_line_chars.sort(key=lambda item: item["centroid"][0])

        if len(current_line_chars) >= min_chars_in_line:
            text_lines.append(current_line_chars)

    for line_chars in text_lines:
        x_coords = np.array([c['centroid'][0] for c in line_chars])
        y_coords = np.array([c['centroid'][1] for c in line_chars])

        if len(x_coords) < curve_degree + 1:
            if len(x_coords) > 1:
                points_np = np.array([(int(p[0]), int(p[1])) for p in line_chars], np.int32)
                cv2.polylines(img_output, [points_np.reshape(-1, 1, 2)], False, (255, 0, 0), 2)
            continue

        poly_coeffs = np.polyfit(x_coords, y_coords, curve_degree)
        poly_func = np.poly1d(poly_coeffs)

        x_min, x_max = x_coords.min(), x_coords.max()
        x_curve = np.linspace(x_min, x_max, 100)
        y_curve = poly_func(x_curve)
        curve_points = np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)], np.int32)
        cv2.polylines(img_output, [curve_points.reshape(-1, 1, 2)], False, (0, 0, 255), 2)

    cv2.imwrite(output_image_path, img_output)
    print(f"Output image with text area curves saved to: {output_image_path}")


if __name__ == "__main__":
    input_image_file = "testt.png"
    output_image_file = "aa.png"
    detect_and_draw_curved_text_area(input_image_file, output_image_file,
                                     max_area=10000,
                                     max_char_dist_x_ratio=1.8,
                                     max_char_dist_y_ratio=0.6,
                                     max_height_diff_ratio=0.25,
                                     curve_degree=2,
                                     min_chars_in_line=4)