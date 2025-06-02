import cv2
import numpy as np

def is_straight_line(x_coords, y_coords, threshold=1.5):
    if len(x_coords) < 2:
        return True
    coeffs = np.polyfit(x_coords, y_coords, deg=1)
    fitted = np.polyval(coeffs, x_coords)
    residuals = np.abs(y_coords - fitted)
    avg_residual = np.mean(residuals)
    return avg_residual < threshold

def analyze_text_region(image_path, region_coords, output_path="output_text_line_analysis.png",
                        min_area=30, max_area=5000, delta=5,
                        curve_degree=2):
    img = cv2.imread(image_path)
    if img is None:
        print("Image could not be loaded.")
        return

    x, y, w, h = region_coords
    roi = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area)
    regions, bboxes = mser.detectRegions(gray)

    char_centroids = []
    filtered_bboxes = []

    for i, bbox in enumerate(bboxes):
        x0, y0, w0, h0 = bbox
        if w0 < 5 or h0 < 5 or w0 > 500 or h0 > 500:
            continue
        aspect_ratio = w0 / float(h0)
        if not (0.1 < aspect_ratio < 10):
            continue
        contour = regions[i]
        if len(contour) < 5:
            continue
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        extent = area / (w0 * h0)
        if not (solidity > 0.3 and extent > 0.15):
            continue
        cx = x0 + w0 // 2
        cy = y0 + h0 // 2
        char_centroids.append((cx, cy))
        filtered_bboxes.append((x0, y0, w0, h0))

    vis = roi.copy()
    for (bx, by, bw, bh) in filtered_bboxes:
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)

    if len(char_centroids) >= 2:
        x_pts = np.array([pt[0] for pt in char_centroids])
        y_pts = np.array([pt[1] for pt in char_centroids])
        if is_straight_line(x_pts, y_pts):
            cv2.line(vis, (x_pts.min(), int(np.mean(y_pts))),
                          (x_pts.max(), int(np.mean(y_pts))), (255, 0, 0), 2)
            print("Detected: Straight line")
        else:
            try:
                poly_coeffs = np.polyfit(x_pts, y_pts, curve_degree)
                poly_func = np.poly1d(poly_coeffs)
                x_fit = np.linspace(x_pts.min(), x_pts.max(), 100)
                y_fit = poly_func(x_fit)
                curve_pts = np.array([[int(xv), int(yv)] for xv, yv in zip(x_fit, y_fit)], dtype=np.int32)
                cv2.polylines(vis, [curve_pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
                print("Detected: Curved line")
            except:
                print("Curve fitting failed.")
    else:
        print("Not enough character points.")

    output_image = img.copy()
    output_image[y:y+h, x:x+w] = vis
    cv2.imwrite(output_path, output_image)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    image_path = "testt.png"
    output_path = "result.png"
    region = (50, 100, 300, 60)  # example, change to your actual coordinates
    analyze_text_region(image_path, region, output_path)