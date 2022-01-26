from iou import calculate_iou
import numpy as np
iou = calculate_iou(np.array([0, 20, 20, 0]), np.array([100, 100, 200, 0]))
print(iou)
 