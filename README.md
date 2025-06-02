import cv2
import numpy as np

def detect_and_draw_curved_text_area(image_path, output_image_path="output_curved_text_areas.png",
                                     min_area=30, max_area=5000, max_variation=0.25,
                                     min_diversity=0.2, delta=5,
                                     max_char_dist_x_ratio=1.5, max_char_dist_y_ratio=0.8,
                                     max_height_diff_ratio=0.3, curve_degree=2, min_chars_in_line=3):
    """
    Detects potential text regions using MSER, applies robust filtering,
    groups them into likely text lines (including curved), and fits a curve to each line.

    Args:
        image_path (str): Path to the input image.
        output_image_path (str): Path to save the output image with detections.
        min_area (int): Minimum area of MSER regions.
        max_area (int): Maximum area of MSER regions.
        max_variation (float): Maximum variation of the MSER regions.
        min_diversity (float): Minimum diversity of the MSER regions.
        delta (int): Delta parameter for MSER.
        max_char_dist_x_ratio (float): Max horizontal distance between char centers
                                       as a multiple of avg char width for grouping.
        max_char_dist_y_ratio (float): Max vertical distance (or misalignment)
                                       as a multiple of avg char height for grouping.
        max_height_diff_ratio (float): Max allowed percentage difference in height
                                        between characters in a line (e.g., 0.3 means 30% difference).
        curve_degree (int): Degree of the polynomial to fit for curves (e.g., 2 for quadratic, 3 for cubic).
        min_chars_in_line (int): Minimum number of characters to consider a valid text line for curve fitting.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'. Please check the path and ensure the file exists.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize MSER detector
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setMaxVariation(max_variation)
    mser.setMinDiversity(min_diversity)
    mser.setDelta(delta)

    regions, bboxes = mser.detectRegions(gray)

    # Filter MSER regions and extract properties for grouping
    candidate_chars = [] # Stores (centroid, width, height, bbox_orig)
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        
        # --- Basic size and aspect ratio filtering ---
        if w < 5 or h < 5 or w > 500 or h > 500: # Filter out very small or very large regions
            continue
        
        aspect_ratio = w / float(h)
        if not (0.05 < aspect_ratio < 15): # Wider range for diverse fonts/styles
            continue

        # --- Contour-based filtering (solidity, extent) ---
        if len(regions[i]) < 5: # Need enough points for meaningful contour properties
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

        # Adjusted thresholds for general text, might need tuning
        if not (solidity > 0.3 and extent > 0.15): 
            continue
        
        # Store character properties
        centroid = (x + w // 2, y + h // 2)
        candidate_chars.append({"centroid": centroid, "w": w, "h": h, "bbox": bbox})

    # Create a copy of the image to draw on
    img_output = img.copy()

    # Draw all filtered MSER bounding boxes in light green for visibility
    for char_data in candidate_chars:
        x, y, w, h = char_data['bbox']
        cv2.rectangle(img_output, (x, y), (x + w, y + h), (100, 255, 100), 1) # Light Green

    # --- Advanced Grouping for Text Lines ---
    # Sort characters by x-coordinate to process roughly left-to-right
    candidate_chars.sort(key=lambda item: item["centroid"][0])

    text_lines = []
    used_indices = set()

    for i, char1 in enumerate(candidate_chars):
        if i in used_indices:
            continue

        current_line_indices = [i]
        current_line_chars = [char1]
        used_indices.add(i)
        
        # Iterate to find subsequent characters that belong to the same line
        for j, char2 in enumerate(candidate_chars):
            if j in used_indices:
                continue
            
            # Use the last character added to the current_line_chars as the reference
            # to check for proximity with new candidates.
            last_char_in_line = current_line_chars[-1]

            # 1. Horizontal Proximity: Is char2 horizontally close to the last char in line?
            avg_width = (last_char_in_line['w'] + char2['w']) / 2
            dist_x = char2['centroid'][0] - last_char_in_line['centroid'][0]
            if not (0 < dist_x < avg_width * max_char_dist_x_ratio):
                continue # Too far or wrong direction

            # 2. Vertical Alignment/Overlap: Do they align vertically?
            # Check if centroids are vertically close, relative to height
            avg_height = (last_char_in_line['h'] + char2['h']) / 2
            dist_y_centroid = abs(last_char_in_line['centroid'][1] - char2['centroid'][1])
            if dist_y_centroid > avg_height * max_char_dist_y_ratio:
                continue # Too much vertical deviation

            # 3. Height Similarity: Are their heights roughly similar?
            height_diff_ratio = abs(last_char_in_line['h'] - char2['h']) / max(last_char_in_line['h'], char2['h'])
            if height_diff_ratio > max_height_diff_ratio:
                continue # Heights too different

            # If all checks pass, add to current line and mark as used
            current_line_indices.append(j)
            current_line_chars.append(char2)
            used_indices.add(j)
            # Re-sort current_line_chars for the next iteration's 'last_char_in_line'
            current_line_chars.sort(key=lambda item: item["centroid"][0])


        if len(current_line_chars) >= min_chars_in_line:
            text_lines.append(current_line_chars)

    # --- Curve Fitting and Drawing ---
    for line_chars in text_lines:
        # Extract centroids for curve fitting
        x_coords = np.array([c['centroid'][0] for c in line_chars])
        y_coords = np.array([c['centroid'][1] for c in line_chars])

        if len(x_coords) < curve_degree + 1:
            # Not enough points for desired polynomial degree, fall back to linear or polyline
            if len(x_coords) > 1:
                points_np = np.array([(int(p[0]), int(p[1])) for p in line_chars], np.int32)
                cv2.polylines(img_output, [points_np.reshape(-1, 1, 2)], False, (255, 0, 0), 2) # Blue, simple polyline
            continue

        # Fit a polynomial curve (e.g., 2nd or 3rd degree)
        # Using polyfit to find coefficients [a, b, c] for y = ax^2 + bx + c
        try:
            poly_coeffs = np.polyfit(x_coords, y_coords, curve_degree)
            poly_func = np.poly1d(poly_coeffs)

            # Generate more points along the fitted curve for smooth drawing
            # Define a finer range of x-values within the min/max x of the text line
            x_min, x_max = x_coords.min(), x_coords.max()
            x_curve = np.linspace(x_min, x_max, 100) # Generate 100 points
            y_curve = poly_func(x_curve)

            # Convert to integer points for OpenCV
            curve_points = np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)], np.int32)

            # Draw the fitted curve in red
            cv2.polylines(img_output, [curve_points.reshape(-1, 1, 2)], False, (0, 0, 255), 2) # Red color

        except np.linalg.LinAlgError:
            # Sometimes polyfit can fail for very few points or specific alignments
            if len(x_coords) > 1:
                points_np = np.array([(int(p['centroid'][0]), int(p['centroid'][1])) for p in line_chars], np.int32)
                cv2.polylines(img_output, [points_np.reshape(-1, 1, 2)], False, (255, 0, 0), 2) # Fallback to blue polyline

    # Save the output image
    cv2.imwrite(output_image_path, img_output)
    print(f"Output image with text area curves saved to: {output_image_path}")

# --- Main execution block ---
if __name__ == "__main__":
    # >>> IMPORTANT: Set the path to your image here <<<
    input_image_file = "testt.png" # Example: "my_document.png" or "path/to/my/image.jpeg"
    output_image_file = "aa.png"

    try:
        detect_and_draw_curved_text_area(input_image_file, output_image_file,
                                         max_area=10000, # Increased max_area as text characters can be larger
                                         max_char_dist_x_ratio=1.8, # More forgiving for spacing variations
                                         max_char_dist_y_ratio=0.6, # Stricter vertical alignment
                                         max_height_diff_ratio=0.25, # Stricter height similarity
                                         curve_degree=2, # Start with quadratic, try 3 for more complex curves
                                         min_chars_in_line=4 # Require more characters for a confident line
                                        )
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have OpenCV (opencv-python) and NumPy installed.")
        print("You can install them using: pip install opencv-python numpy")
