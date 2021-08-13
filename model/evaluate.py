from pdb import set_trace as bp
import sys
import os
from os import walk
import numpy as np
import cv2

# check input arguments
if len(sys.argv) != 2:
    print('Incorrect number of input arguments.')
    print('Please input results directory')
    exit()
else:
    results_dir = sys.argv[1]

# get list of files
pred_path = os.path.join(results_dir, 'pred')
gt_path = os.path.join(results_dir, 'gt')
_, _, pred_files = next(walk(pred_path))
_, _, gt_files = next(walk(gt_path))
pred_files = np.sort(np.array(pred_files))
gt_files = np.sort(np.array(gt_files))

# compute IOU
def get_iou(pred, gt, color):
    pred_mask = (pred[:,:,0] == color[0]) & (pred[:,:,1] == color[1]) & (pred[:,:,2] == color[2])
    gt_mask = (gt[:,:,0] == color[0]) & (gt[:,:,1] == color[1]) & (gt[:,:,2] == color[2])
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    gt_sum = np.sum(gt_mask)
    ni = np.sum(intersection)
    nu = np.sum(union)
    if nu == 0:
        iou = 0
    else:
        iou = ni / nu
    return iou, ni, nu, gt_sum, pred_mask, gt_mask

# loop through files
LBL = np.array(['Building', 'Fence', 'Pole', 'Sidewalk', 'Vegetation', 'Wall', 'Road', 'Traffic light', \
                'Person', 'Car', 'Truck', 'Bus', 'Train', 'Bicycle', 'Other'])
N = len(LBL)
NI = np.zeros(N)
NU = np.zeros(N)
W = np.zeros(N)
IOU = np.zeros(N)
for pred_file, gt_file in zip(pred_files, gt_files):
    # load images
    pred = cv2.imread(os.path.join(pred_path, pred_file))
    gt = cv2.imread(os.path.join(gt_path, gt_file))
    
    # check shape
    assert(pred.shape == gt.shape)

    # reshape to standard shape
    w = 1024
    h = 512
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    # initialize masks
    PRED_MASK = np.zeros((pred.shape)[:2]).astype(bool)
    GT_MASK = np.zeros((gt.shape)[:2]).astype(bool)

    # class 0
    i = 0
    color = [70,70,70]
    iou, ni, nu, w, pred_mask, gt_mask = get_iou(pred, gt, color)
    NI[i] += ni
    NU[i] += nu
    W[i] += w
    PRED_MASK = PRED_MASK | pred_mask
    GT_MASK = GT_MASK | gt_mask

    # class 1
    i = 1
    color = [153,153,190]
    iou, ni, nu, w, pred_mask, gt_mask = get_iou(pred, gt, color)
    NI[i] += ni
    NU[i] += nu
    W[i] += w
    PRED_MASK = PRED_MASK | pred_mask
    GT_MASK = GT_MASK | gt_mask

    # class 2
    i = 2
    color = [153,153,153]
    iou, ni, nu, w, pred_mask, gt_mask = get_iou(pred, gt, color)
    NI[i] += ni
    NU[i] += nu
    W[i] += w
    PRED_MASK = PRED_MASK | pred_mask
    GT_MASK = GT_MASK | gt_mask

    # class 3
    i = 3
    color = [232,35,244]
    iou, ni, nu, w, pred_mask, gt_mask = get_iou(pred, gt, color)
    NI[i] += ni
    NU[i] += nu
    W[i] += w
    PRED_MASK = PRED_MASK | pred_mask
    GT_MASK = GT_MASK | gt_mask

    # class 4
    i = 4
    color = [35,142,107]
    iou, ni, nu, w, pred_mask, gt_mask = get_iou(pred, gt, color)
    NI[i] += ni
    NU[i] += nu
    W[i] += w
    PRED_MASK = PRED_MASK | pred_mask
    GT_MASK = GT_MASK | gt_mask

    # class 5
    i = 5
    color = [156,102,102]
    iou, ni, nu, w, pred_mask, gt_mask = get_iou(pred, gt, color)
    NI[i] += ni
    NU[i] += nu
    W[i] += w
    PRED_MASK = PRED_MASK | pred_mask
    GT_MASK = GT_MASK | gt_mask

    # class 6
    i = 6
    color = [128,64,128]
    iou1, ni1, nu1, w1, pred_mask1, gt_mask1 = get_iou(pred, gt, color)
    color = [50,234,157]
    iou2, ni2, nu2, w2, pred_mask2, gt_mask2 = get_iou(pred, gt, color)
    NI[i] += (ni1 + ni2)
    NU[i] += (nu1 + nu2)
    W[i] += (w1 + w2)
    PRED_MASK = PRED_MASK | pred_mask1
    PRED_MASK = PRED_MASK | pred_mask2
    GT_MASK = GT_MASK | gt_mask1
    GT_MASK = GT_MASK | gt_mask2

    # class 7
    i = 7
    color = [30,170,250]
    iou1, ni1, nu1, w1, pred_mask1, gt_mask1 = get_iou(pred, gt, color)
    color = [0,220,220]
    iou2, ni2, nu2, w2, pred_mask2, gt_mask2 = get_iou(pred, gt, color)
    NI[i] += (ni1 + ni2)
    NU[i] += (nu1 + nu2)
    W[i] += (w1 + w2)
    PRED_MASK = PRED_MASK | pred_mask1
    PRED_MASK = PRED_MASK | pred_mask2
    GT_MASK = GT_MASK | gt_mask1
    GT_MASK = GT_MASK | gt_mask2

    # class 8
    i = 8
    color = [60,20,220]
    iou1, ni1, nu1, w1, pred_mask1, gt_mask1 = get_iou(pred, gt, color)
    color = [0,0,255]
    iou2, ni2, nu2, w2, pred_mask2, gt_mask2 = get_iou(pred, gt, color)
    NI[i] += (ni1 + ni2)
    NU[i] += (nu1 + nu2)
    W[i] += (w1 + w2)
    PRED_MASK = PRED_MASK | pred_mask1
    PRED_MASK = PRED_MASK | pred_mask2
    GT_MASK = GT_MASK | gt_mask1
    GT_MASK = GT_MASK | gt_mask2

    # class 9
    i = 9
    color = [142,0,0]
    iou, ni, nu, w, pred_mask, gt_mask = get_iou(pred, gt, color)
    NI[i] += ni
    NU[i] += nu
    W[i] += w
    PRED_MASK = PRED_MASK | pred_mask
    GT_MASK = GT_MASK | gt_mask

    # class 10
    i = 10
    color = [70,0,0]
    iou, ni, nu, w, pred_mask, gt_mask = get_iou(pred, gt, color)
    NI[i] += ni
    NU[i] += nu
    W[i] += w
    PRED_MASK = PRED_MASK | pred_mask
    GT_MASK = GT_MASK | gt_mask

    # class 11
    i = 11
    color = [100,60,0]
    iou, ni, nu, w, pred_mask, gt_mask = get_iou(pred, gt, color)
    NI[i] += ni
    NU[i] += nu
    W[i] += w
    PRED_MASK = PRED_MASK | pred_mask
    GT_MASK = GT_MASK | gt_mask

    # class 12
    i = 12
    color = [100,80,0]
    iou, ni, nu, w, pred_mask, gt_mask = get_iou(pred, gt, color)
    NI[i] += ni
    NU[i] += nu
    W[i] += w
    PRED_MASK = PRED_MASK | pred_mask
    GT_MASK = GT_MASK | gt_mask

    # class 13
    i = 13
    color = [230,0,0]
    iou1, ni1, nu1, w1, pred_mask1, gt_mask1 = get_iou(pred, gt, color)
    color = [32,11,119]
    iou2, ni2, nu2, w2, pred_mask2, gt_mask2 = get_iou(pred, gt, color)
    NI[i] += (ni1 + ni2)
    NU[i] += (nu1 + nu2)
    W[i] += (w1 + w2)
    PRED_MASK = PRED_MASK | pred_mask1
    PRED_MASK = PRED_MASK | pred_mask2
    GT_MASK = GT_MASK | gt_mask1
    GT_MASK = GT_MASK | gt_mask2

    # class 14
    i = 14
    pred_mask = ~PRED_MASK
    gt_mask = ~GT_MASK
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    w = np.sum(gt_mask)
    ni = np.sum(intersection)
    nu = np.sum(union)
    NI[i] += ni
    NU[i] += nu
    W[i] += w

# print IOUs
for i in range(N):
    if NU[i] == 0:
        IOU[i] = 0
        print('{}: ---'.format(LBL[i]))
    else:
        IOU[i] = NI[i] / NU[i]
        print('{}: {:.3f}'.format(LBL[i], IOU[i]))
print('------------------')

# mean IOU
print('mIOU: {:.3f}'.format(np.mean(IOU[IOU != 0])))

# weighted mean IOU
w_total = np.sum(W)
W /= w_total
print('wMIOU: {:.3f}'.format(np.sum(W*IOU)))
