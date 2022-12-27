import cv2
import numpy as np

def show_spec_area(image, bbox, bg_val):
    x0, y0, x1, y1 = bbox
    img = image.copy()
    img[:x0, :] = bg_val
    img[x1:, :] = bg_val
    img[:, :y0] = bg_val
    img[:, y1:] = bg_val
    return img

def crop_a(img):
    px, py = np.where(img!=255)
    px.sort()
    py.sort()
    x0, x1, y0, y1 = px[0]+1, px[-1], py[0]+1, py[-1]
    return x0, x1, y0, y1

def clean_box_num(image):
    img = image.copy()
    x, y = img.shape[:2]
    low_idxs, y_sum = [], []
    for i in range(x):
        y_sum.append(sum(img[i, :])/y)
        if y_sum[-1] < 100:
            low_idxs.append(i)
            
    for low_idx in low_idxs:
        img[low_idx:low_idx+40, :] = 255
    
    return img

def calc_bottom_pt(image):
    the_cls_bottom_pts = []
    img = 255*(image>0)
    x, y = img.shape[:2]
    
    for i in range(x):
        if sum(img[i, :])/y < 150:
            for j in range(1, y):
                if img[i, j-1] and not img[i, j]:
                    j1 = j
                elif not img[i, j-1] and img[i, j]:
                    the_cls_bottom_pts.append([i, (j+j1)//2])
                    j1 = 0
    return np.array(the_cls_bottom_pts)

def get_cls_from_pt(the_cls_bottom_pts, bbox):
    cur_pt = (bbox[2], (bbox[1]+bbox[3])//2)
    dist = [np.linalg.norm(pt-cur_pt) for pt in the_cls_bottom_pts]
    return np.argmin(dist)+1

def threshold_segm(image, threshold=254):
    img = image.copy()
    img = clean_box_num(img)
    img[img==0] = 255
    
    return 255*(img<threshold).astype(np.uint8)
    
def fill_holes(mask):
    im = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    im_fill = im.copy()
    h, w = im.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_fill, mask, (0, 0), 255)
    im_fill_inv = cv2.bitwise_not(im_fill)
    im_out = im | im_fill_inv
    return im_out[1:-1, 1:-1]
