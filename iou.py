import numpy as np

from utils import get_data, check_results


def calculate_ious(gt_bboxes, pred_bboxes):
    """
    calculate ious between 2 sets of bboxes 
    args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 pred array
    returns:
    - iou [array]: NxM array of ious
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i,j] = calculate_iou(gt_bbox, pred_bbox)
    return ious


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    ## IMPLEMENT THIS FUNCTION
    overlapping_area = get_overlapping(gt_bbox, pred_bbox)
    prediction_area = get_area(*pred_bbox)
    gt_area = get_area(*gt_bbox)
    total_area = prediction_area + gt_area - overlapping_area
    iou = overlapping_area/total_area
    return iou

def get_overlapping(gt_bbox, pred_bbox):
    x1, y1, x2, y2 = gt_bbox
    x3, y3, x4, y4 = pred_bbox
    a = max(x1, x3)
    b = max(y1, y3)
    c = min(x2, x4)
    d = min(y2, y4)
    return get_area(a, b, c, d)

def get_area (x1, y1, x2, y2):
    # where `x1 < x2` and `y1 < y2`. 
    # `(x, y1)` are the coordinates of the upper left corner 
    # `(x2, y2)` the coordinates of the lower right corner of the bounding box.
    if x2 - x1 < 0 or y2 - y1 < 0:
        return 0
    return (x2 - x1) * (y2 - y1)

if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    
    ious = calculate_ious(gt_bboxes, pred_boxes)
    check_results(ious)