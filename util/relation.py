import numpy as np

def spatial_relation(a, b, w, h):
    """
    Calculate the spatial relation between 2 objects.
    Input:
        a: bbox of object i
        b: bbox of object j
        w: the width of the whole image
        h: the height of the whole image
    Output: the type number of both a and b
    The definition of spatial relations <a-b> is described in 'Exploring Visual Relationship for Image Captioning'
        1: a is inside b
        2: a is coverd by b
        3: a overlaps b (IoU >= 0.5)
        4~11: index = ceil(delta / 45) + 3 (IoU < 0.5)
    """
    # get IoU region box
    iou = np.array([
        max(a[0], b[0]), max(a[1], b[1]), # (x0, y0)
        min(a[2], b[2]), min(a[3], b[3])  # (x1, y1)
    ])

    if np.array_equal(iou, b): return 1, 2 # If IoU == b: b is inside a
    elif np.array_equal(iou, a): return 2, 1 # Else if IoU == a: a is covered by b

    # Else if IoU >=0.5: a and b overlap
    area = lambda x: (x[3]-x[1])*(x[2]-x[0])
    iou =  area(iou) / (area(a) + area(b) - area(iou))
    if iou >= 0.5: return 3, 3

    # Else if the ratio of the relation distance and the diagonal length 
    # of the whole image is less than 0.5: compute the angle between a and b
    center = lambda x: np.array([x[0]+(x[2]-x[0])/2, x[1]+(x[3]-x[1])/2])
    a = center(a)
    b = center(b)
    dist = np.linalg.norm(a-b) / np.linalg.norm([w,h])
    if dist <= 0.5:
        a = b-a
        delta = np.rad2deg(np.arctan2(*a)) - 90
        index = lambda x: int(np.ceil((x % 360) / 45) + 3)
        return index(delta), index(delta+180)
    
    # Else: no relation between a and b
    return 0, 0


def semantic_relation(a, b, w, h):
    """
    Calculate the semantic relation between 2 objects.
    Input:
        a: bbox of object i
        b: bbox of object j
        w: the width of the whole image
        h: the height of the whole image
    Output: the type number of both a and b
    The definition of semantic relations <a-b> is described in 'Exploring Visual Relationship for Image Captioning'
    """
    #########################
    # TODO: semantic relation
    #########################
    return


def relation_graph(bbox, w, h, relation=spatial_relation):
    """Build relation graph of objects in an image.
    Input:
        bbox: [num_objs, 4] the bounding boxes of objects in an image
        w: the width of the image
        h: the height of the image
        relation: function to construct relation graph (default = spatial_relation)
    Output: [num_objs, num_objs]
    """
    num_objs = bbox.shape[0]
    output = np.zeros((num_objs, num_objs))

    for i in range(num_objs):
        for j in range(i+1, num_objs):
            output[i,j], output[j,i] = relation(bbox[i], bbox[j], w, h)
    return output