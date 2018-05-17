# Author: Alexander Decurnou

import collections
import cv2
import math
import numpy as np
from scipy import linalg
import xml.etree.ElementTree as ET


def bounding_boxes(jpg_file, xml_file):
    # lists to hold xmin, xmax, ymin, ymax coordinates
    boxes = []

    # convert image to array
    img = cv2.imread(jpg_file)
    # convert the color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # parse xml for bounding box coordinates
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # for every bounding box in the image, add the max and min coordinates
    # for the x, y points into a tuple and at that tuple to the box list
    for member in root.findall('object'):
        xmin = int(member[4][0].text)
        xmax = int(member[4][2].text)
        ymin = int(member[4][1].text)
        ymax = int(member[4][3].text)
        boxes.append((xmin, xmax, ymin, ymax))

    print("Amount of boxes: ", len(boxes))

    return img, boxes


def ground_truth_saliency_map(image, image_height, image_width, stride, boxes):
    """
        Gnerates a ground truth saliency map for use in training by creating
        a Gaussian distribution given bounding box location and peaks.
            iamge_height:   (Integer)   height of ground truth image
            image_width:    (Integer)   width of ground truth image
            stride:         (Integer)   stride of the network
            boxes:          (List)      list of bounding boxes
    """

    map_width = math.floor(image_width / stride)
    map_height = math.floor(image_height / stride)

    gt_sal_map = np.zeros((map_height, map_width))
    print(gt_sal_map)

    # TODO: call the first element of each of the coordinate lists, then the
    # second element, and so on
    for x in range(map_width):
        for y in range(map_height):
            gt_sal_map[x][y] = x_y_element(x, y, boxes, stride)

    return image, gt_sal_map


def x_y_element(x, y, boxes, stride):
    """
        Calculates a value for the Gaussian distribution at an element (x, y)
        on a 2D array given a box and the network stride.
            boxes:          (List)      list of bounding boxes
            stride:         (Integer)   stride of the network
    """

    x_y_element = 0

    for i, box in enumerate(boxes, 1):

        print("This is Box #:", i)

        box_center_x = box[1] - box[0] / 2
        box_center_y = box[3] - box[2] / 2
        box_width = box[1] - box[0]
        box_height = box[3] - box[2]

        v_xy = np.matrix([x, y]).T
        u_i = np.matrix([math.floor(box_center_x / stride),
                        math.floor(box_center_y / stride)]).T

        power = (v_xy - u_i).T @ covariance(box_width, box_height, stride)
        power = power @ (v_xy - u_i)
        power = power * -0.5
        print("power: ", power)

        # Figure out how to test if v_xy is in roi
        # if (v_x > xmin and v_x < xmax) and (v_y > ymin and v_y < ymax)
        if (v_xy[0] * stride > box[0] and v_xy[0] * stride < box[1]) and (v_xy[1] * stride > box[2] and v_xy[1] * stride < box[3]):
            indicator = 1
        else:
            indicator = 0

        print("indicator: ", indicator)

        op = linalg.expm(power) * indicator

        print("op: ", op)

        x_y_element += op

        print("XY Element: ", x_y_element)

    return x_y_element


def covariance(width, height, stride):
    """
        Calculate the covariance of a box given the width, height, and stride.
            width:          (Integer)   width of the bounding box
            height:         (Integer)   height of the bounding box
            stride:         (Integer)   stride of the network
    """

    covariance = np.matrix([[math.pow((math.floor(width / stride)), 2) / 4, 0],
                           [0, math.pow((math.floor(height / stride)), 2) / 4]])

    return covariance


def single_box_gen(sal_map, confidence_threshold=0.7):
    """
        If the subitizing branch returns 1, then there is just a single salient
        object in the original image. In this case, this algorithm takes the
        image, detects the contours in it, and generates bounding boxes to
        enclose the object. The box with biggest region of interest is selected
        as the bounding box of that single salient object.
            sal_map:                (Array) saliency map of the original image
            confidence_threshold:   (Float) cut-off for image thresholding
    """

    # make a tuple to hold box coordinates
    Box = collections.namedtuple('Box', ['x', 'y', 'w', 'h'])
    max_box = Box(0, 0, 0, 0)
    bounding_boxes = set()
    max_roi = 0

    # threshold step: generate a binary version of the saliency map
    sal_map_gray = cv2.cvtColor(sal_map, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(sal_map_gray, math.floor(255 * confidence_threshold), 255, cv2.THRESH_BINARY)

    # contour detection step: uses Teh-Chin chain approximation
    mod_sal_map_gray, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # box generation step
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bounding_box = Box(x, y, w, h)
        bounding_boxes |= bounding_box

    for box in bounding_boxes:
        x = getattr(box, x)
        y = getattr(box, y)
        w = getattr(box, w)
        h = getattr(box, h)
        roi = sal_map[y:y+h, x:x+w]
        if roi > max_roi:
            max_box = box

    return max_box


def multi_box_gen(sal_map, n_sub, confidence_threshold=0.7):
    """
        If the subitizing branch returns n_sub greater than 1, then this
        algorithm takes the saliency map and through a series of decreasing
        thresholds, it generates peaks and adds them to a set. The set of peaks
        is then filtered by the confidence threshold. Separating lines are then
        found between each peak and those lines are used to create bounding
        boxes.
            sal_map:                (Array) saliency map of the original image
            n_sub:                  (Integer) predicted number of objects from subitizing branch
            confidence_threshold:   (Float) cut-off for image thresholding
    """

    peak_detection_thresholds = [0.95, 0.9, 0.8, 0.6]

    peaks = set()
    boxes = set()

    while len(peaks) < n_sub:
        for threshold in peak_detection_thresholds:
            sal_map_gray = cv2.cvtColor(sal_map, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(sal_map_gray, 255 * threshold, 255, cv2.THRESH_BINARY)

            mod_sal_map_gray, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

            # Add the arguments of the contours into the set of peaks
            for c in contours:
                peaks |= c[0][0]

    # filter out peaks from the set that don't meet the confidence threshold
    for peak in peaks:
        if sal_map[peak[0]][peak[1]] < confidence_threshold:
            peaks -= peak

    # find separating lines
    vert_lines = []
    horiz_lines = []
    peak_list = list(peaks)

    # iterate through each pair of peaks
    for i, pi in enumerate(peak_list):
        for pj in peak_list[i+1:]:

            # find the min and max locations for the peaks for the range
            min_x, min_y = min(pi[0], pj[0]), min(pi[1], pj[1])
            max_x, max_y = max(pi[0], pj[0]), max(pi[1], pj[1])

            height = len(sal_map[0])
            width = len(sal_map)

            # create a list of all the vertical and horizontal lines
            # find the ones with the max values across that list
            # and then take the minimum of values in the range between x and y
            vert_min, vert_x = min([(max([sal_map[x][y] for y in range(width)]), x) for x in range(min_x, max_x + 1)])
            horiz_min, horiz_y = min([(max([sal_map[x][y] for x in range(height)]), y) for y in range(min_y, max_y + 1)])

            if vert_min < horiz_min:
                vert_lines.append(vert_x)
            else:
                horiz_lines.append(horiz_y)

    vert_lines.sort()
    horiz_lines.sort()

    for peak in peaks:

        # peak has the form [x, y]

        # set up the default values for the lines
        left = 0
        right = len(sal_map)
        top = 0
        bottom = len(sal_map[0])

        # the left line should be as close as, but not equal to, the
        # x-value as possible, and the same goes for the right line
        for line in vert_lines:
            if line < peak[0] and line > left:
                left = line
            elif line > peak[0] and line < right:
                right = line

        # the top line should be as close as, but not equal to, the
        # y-value as possible, and the same goes for the bottom line
        for line in horiz_lines:
            if line < peak[1] and line > top:
                top = line
            elif line > peak[1] and line < bottom:
                bottom = line

        # the ROI should be an area enclosed by all of the boundary lines
        roi = sal_map[top:bottom, left:right]

        # pass the ROI to the single box generator
        # function and add to the set of boxes
        boxes |= single_box_gen(roi)

    return boxes
