import numpy as np

def spatial_relation(a, b):
    """
    Calculate the spatial relation between 2 bounding boxes, and return the type number of relation.
    The definition of types of spatial relations <a-b> is described in "Exploring Visual Relationship for Image Captioning".
        1: a is inside b
        2: a is coverd by b
        3: a overlaps b (IoU >= 0.5)
        4~11: index = ceil(delta / 45) + 3 (IoU < 0.5)
    Output: the type number of both a and b
    """
    def area(x): return (x[3]-x[1])*(x[2]-x[0])
    
    # get IoU region box
    iou_box = np.array([
        max(a[0], b[0]), max(a[1], b[1]), # (x0, y0)
        min(a[2], b[2]), min(a[3], b[3])  # (x1, y1)
    ])

    if iou_box == b: return 1, 2 # If IoU == b: b is inside a
    elif iou_box == a: return 2, 1 # Else if IoU == a: a is covered by b

    # Else if IoU >=0.5: a and b overlap
    iou =  area(iou_box) / (area(a) + area(b) - area(iou_box))
    if iou >= 0.5: return 3, 3

    # Else if the ratio of the relation distance and the diagonal length of the whole image is less than 0.5:
    # compute the angle between a and b
    ratio = 0
    # TODO: compute the ratio
    if ratio >= 0.5:
        index = np.ceil(np.angle(b-a) / 45) + 3
        return index, index
    
    # Else: no relation between a and b
    return 0, 0