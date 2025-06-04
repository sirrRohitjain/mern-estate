=====================

STEP 0: Install Requirements

=====================

pip install torch torchvision

pip install opencv-python numpy matplotlib scikit-learn

pip install git+https://github.com/clovaai/CRAFT-pytorch.git

import cv2 import torch import numpy as np from craft import CRAFT from craft_utils import getDetBoxes from imgproc import resize_aspect_ratio, normalizeMeanVariance from skimage import io from sklearn.cluster import DBSCAN from numpy.polynomial.polynomial import Polynomial

=====================

STEP 1: Load CRAFT model

=====================

def load_craft_model(): net = CRAFT() net.load_state_dict(torch.load('craft_mlt_25k.pth', map_location='cpu')) net.eval() return net

=====================

STEP 2: Detect words/characters using CRAFT

=====================

def detect_text_regions(image, net): # Resize & normalize img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 1280) x = normalizeMeanVariance(img_resized) x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()

# Forward
with torch.no_grad():
    y, _ = net(x)

score_text = y[0, :, :, 0].cpu().data.numpy()
score_link = y[0, :, :, 1].cpu().data.numpy()

boxes, polys = getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4, False)
ratio_h = image.shape[0] / float(size_heatmap[0])
ratio_w = image.shape[1] / float(size_heatmap[1])
boxes = np.array(boxes)
boxes *= [ratio_w, ratio_h]
return boxes.astype(int)

=====================

STEP 3: Group characters into words (if needed)

=====================

def group_boxes_dbscan(boxes): # Use box centers for clustering centers = np.array([np.mean(box, axis=0) for box in boxes]) clustering = DBSCAN(eps=50, min_samples=2).fit(centers) return clustering.labels_, centers

=====================

STEP 4: Determine if text is curved or straight

=====================

def fit_and_decide_curve(centroids, degree=2): x = centroids[:, 0] y = centroids[:, 1] if len(x) < degree + 1: return None, 'straight' p = Polynomial.fit(x, y, deg=degree) y_fit = p(x) residuals = np.abs(y - y_fit) is_curved = np.std(residuals) > 3 return p.convert().coef, 'curved' if is_curved else 'straight'

=====================

STEP 5: Draw curve or line

=====================

def draw_curve_or_line(image, centroids, fit_type='straight', coef=None): img = image.copy() x = np.array(sorted([int(pt[0]) for pt in centroids])) if coef is not None: if fit_type == 'straight': y = coef[0] + coef[1] * x else: y = coef[0] + coef[1] * x + coef[2] * x**2 pts = np.vstack([x, y]).T.astype(int) for pt1, pt2 in zip(pts, pts[1:]): cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 0), 2) return img

=====================

STEP 6: Run everything

=====================

def process_image(image_path): net = load_craft_model() image = cv2.imread(image_path) orig = image.copy() boxes = detect_text_regions(image, net)

labels, centers = group_boxes_dbscan(boxes)

for label in np.unique(labels):
    if label == -1:  # noise
        continue
    group = centers[labels == label]
    coef, curve_type = fit_and_decide_curve(group, degree=2)
    image = draw_curve_or_line(image, group, curve_type, coef)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

Example usage

process_image("your_image.jpg")

