# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def midp(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    midX = (x1 + x2) // 2
    midY = (y1 + y2) // 2
    return [midX, midY]

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)
                print(poly)
                # poly = poly.reshape(-1, 2)
                # !  -----------------------------
                # !  changes
                # !  -----------------------------
                coords = poly
                n = len(coords) // 2
                upper = coords[0:n:1]
                lower = coords[n::]
                pointsU = np.array(upper, dtype=np.int32).reshape(-1, 2)
                pointsL = np.array(lower, dtype=np.int32).reshape(-1, 2)
                pointsL = pointsL[::-1]
                mid = [midp(p1, p2) for p1, p2 in zip(pointsU, pointsL)]
                mid = np.array(mid)

                x = mid[:, 0]
                y = mid[:, 1]
                coeffs = np.polyfit(x, y, deg=2)
                poly = np.poly1d(coeffs)
                x_smooth = np.linspace(np.min(x), np.max(x), 200)
                y_smooth = poly(x_smooth)
                curve_pts = np.stack((x_smooth, y_smooth), axis=-1).astype(np.int32)
                curve_pts = curve_pts.reshape(-1, 1, 2)
                cv2.polylines(img, [curve_pts], False, (255, 0, 0), thickness=2)

                # !  -----------------------------


                # cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(255, 255, 255), thickness=2)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # Save result image
        cv2.imwrite(res_img_file, img)