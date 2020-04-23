import numpy as np
import random
import torch


def set_all_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def visualize_bbox(img_path, target, plot_objects=True, plot_parts=True, out_img_path='bbox_viz.jpg'):
    """
    Required library: https://github.com/nalepae/bounding-box/
    """
    import cv2
    from bounding_box import bounding_box as bb

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    for obj in target['object']:
        if plot_objects:
            xmin = obj['bndbox']['xmin']
            ymin = obj['bndbox']['ymin']
            xmax = obj['bndbox']['xmax']
            ymax = obj['bndbox']['ymax']
            bb.add(img, xmin, ymin, xmax, ymax, obj['name'])
        if plot_parts:
            for part in obj['parts']:
                xmin = part['bndbox']['xmin']
                ymin = part['bndbox']['ymin']
                xmax = part['bndbox']['xmax']
                ymax = part['bndbox']['ymax']
                bb.add(img, xmin, ymin, xmax, ymax, part['name'])
    
    cv2.imwrite(out_img_path, img)
    cv2.imshow(target['filename'], img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

