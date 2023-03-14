import numpy as np


def iou(bbox1: list, bbox2: list) -> float:
    (x1, w1, y1, h1) = bbox1
    (x2, w2, y2, h2) = bbox2
    squad = np.zeros((max((x1 + w1), (x2 + w2)), max((y1 + h1), (y2 + h2))))
    squad[x1:w1, y1:h1] = 1
    lenSquad1 = len(np.where(squad == 1)[0])
    squad[x2:w2, y2:h2] += 1
    lenSquad2 = len(np.where(squad == 2)[0])
    if lenSquad1 == 0 and lenSquad2 == 0:
        return 0
    elif lenSquad1 == 0:
        return 1
    else:
        return lenSquad2 / (w2*h2)


bbox1 = [0, 10, 0, 10]
bbox2 = [0, 10, 1, 10]
bbox3 = [20, 30, 20, 30]
bbox4 = [5, 15, 5, 15]
assert iou(bbox1, bbox1) == 1.0
assert iou(bbox1, bbox2) == 0.9
assert iou(bbox1, bbox3) == 0.0
