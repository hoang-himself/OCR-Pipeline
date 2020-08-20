import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from craft_utils import adjustResultCoordinates, getDetBoxes, getDetBoxes_core, getPoly_core
from imgproc import *
from file_utils import *
import json
import zipfile

from craft import CRAFT
from collections import OrderedDict

text_threshold = 0.7
low_text = 0.4
link_threshold = 0.4
# cuda = True
cuda = False
canvas_size = 1280
mag_ratio = 1.5
trained_model_path = './craft_mlt_25k.pth'
poly = False
refine = False
show_time = False
refine_net = None


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR,
                                                                  mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = cvt2HeatmapImg(render_img)

    if show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


net = CRAFT()
net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location='cpu')))
net.eval()

# image_path = './doc/2.jpg'
# image = loadImage(image_path)
# bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net)
#
# poly_indexes = {}
# central_poly_indexes = []
# for i in range(len(polys)):
#     poly_indexes[i] = polys[i]
#     x_central = (polys[i][0][0] + polys[i][1][0] + polys[i][2][0] + polys[i][3][0]) / 4
#     y_central = (polys[i][0][1] + polys[i][1][1] + polys[i][2][1] + polys[i][3][1]) / 4
#     central_poly_indexes.append({i: [int(x_central), int(y_central)]})

import copy
import math

import copy

UNCLASSIFIED = -2
NOISE = -1

UNCLASSIFIED = -2


class Point:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.cluster_id = UNCLASSIFIED

    def __repr__(self):
        return '(x:{}, y:{}, id:{}, cluster:{})' \
            .format(self.x, self.y, self.id, self.cluster_id)


def n_pred(p1, p2):
    return (p1.x - p2.x) ** 2 / 160000 + (p1.y - p2.y) ** 2 / 225 <= 1


def w_card(points):
    return len(points)


def GDBSCAN(points, n_pred, min_card, w_card):
    points = copy.deepcopy(points)
    cluster_id = 0
    for point in points:
        if point.cluster_id == UNCLASSIFIED:
            if _expand_cluster(points, point, cluster_id, n_pred, min_card,
                               w_card):
                cluster_id = cluster_id + 1
    clusters = {}
    for point in points:
        key = point.cluster_id
        if key in clusters:
            clusters[key].append(point)
        else:
            clusters[key] = [point]
    return list(clusters.values())


def _expand_cluster(points, point, cluster_id, n_pred, min_card, w_card):
    if not _in_selection(w_card, point):
        points.change_cluster_id(point, UNCLASSIFIED)
        return False

    seeds = points.neighborhood(point, n_pred)
    if not _core_point(w_card, min_card, seeds):
        points.change_cluster_id(point, NOISE)
        return False

    points.change_cluster_ids(seeds, cluster_id)
    seeds.remove(point)

    while len(seeds) > 0:
        current_point = seeds[0]
        result = points.neighborhood(current_point, n_pred)
        if w_card(result) >= min_card:
            for p in result:
                if w_card([p]) > 0 and p.cluster_id in [UNCLASSIFIED, NOISE]:
                    if p.cluster_id == UNCLASSIFIED:
                        seeds.append(p)
                    points.change_cluster_id(p, cluster_id)
        seeds.remove(current_point)
    return True


def _in_selection(w_card, point):
    return w_card([point]) > 0


def _core_point(w_card, min_card, points):
    return w_card(points) >= min_card


class Points:
    def __init__(self, points):
        self.points = points

    def __iter__(self):
        for point in self.points:
            yield point

    def __repr__(self):
        return str(self.points)

    def get(self, index):
        return self.points[index]

    def neighborhood(self, point, n_pred):
        return list(filter(lambda x: n_pred(point, x), self.points))

    def change_cluster_ids(self, points, value):
        for point in points:
            self.change_cluster_id(point, value)

    def change_cluster_id(self, point, value):
        index = (self.points).index(point)
        self.points[index].cluster_id = value

    def labels(self):
        return set(map(lambda x: x.cluster_id, self.points))


# X = []
#
# for idx, x in enumerate(central_poly_indexes):
#     point = Point(x[idx][0], x[idx][1], idx)
#     X.append(point)
# clustered = GDBSCAN(Points(X), n_pred, 1, w_card)
# print(clustered)
#
# cluster_values = []
# for cluster in clustered:
#     sort_cluster = sorted(cluster, key=lambda elem: (elem.x, elem.y))
#     print(sort_cluster)
#     max_point_id = sort_cluster[len(sort_cluster) - 1].id
#     min_point_id = sort_cluster[0].id
#     print(poly_indexes.get(min_point_id))
#     max_rectangle = sorted(poly_indexes[max_point_id], key=lambda elem: (elem[0], elem[1]))
#     min_rectangle = sorted(poly_indexes[min_point_id], key=lambda elem: (elem[0], elem[1]))
#
#     right_above_max_vertex = max_rectangle[len(max_rectangle) - 1]
#     right_below_max_vertex = max_rectangle[len(max_rectangle) - 2]
#     left_above_min_vertex = min_rectangle[0]
#     left_below_min_vertex = min_rectangle[1]
#     cluster_values.append(
#         [left_above_min_vertex, left_below_min_vertex, right_above_max_vertex, right_below_max_vertex])

import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
# files = ['./doc/' + f for f in listdir('./doc') if isfile(join('./doc', f))]
files = ['./QD_image/binary_page/' + f for f in listdir('./QD_image/binary_page/') if isfile(join('./QD_image/binary_page/', f))]
image_index = 0
for file_path in files:
    print('------------------'+file_path)
    print(image_index)
    image = loadImage(file_path)
    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net)

    poly_indexes = {}
    central_poly_indexes = []
    for i in range(len(polys)):
        poly_indexes[i] = polys[i]
        x_central = (polys[i][0][0] + polys[i][1][0] + polys[i][2][0] + polys[i][3][0]) / 4
        y_central = (polys[i][0][1] + polys[i][1][1] + polys[i][2][1] + polys[i][3][1]) / 4
        central_poly_indexes.append({i: [int(x_central), int(y_central)]})

    X = []
    for idx, x in enumerate(central_poly_indexes):
        point = Point(x[idx][0], x[idx][1], idx)
        X.append(point)
    clustered = GDBSCAN(Points(X), n_pred, 1, w_card)
    # print(clustered)

    cluster_values = []
    for cluster in clustered:
        sort_cluster = sorted(cluster, key=lambda elem: (elem.x, elem.y))
        # print(sort_cluster)
        max_point_id = sort_cluster[len(sort_cluster) - 1].id
        min_point_id = sort_cluster[0].id
        # print(poly_indexes.get(min_point_id))
        max_rectangle = sorted(poly_indexes[max_point_id], key=lambda elem: (elem[0], elem[1]))
        min_rectangle = sorted(poly_indexes[min_point_id], key=lambda elem: (elem[0], elem[1]))

        right_above_max_vertex = max_rectangle[len(max_rectangle) - 1]
        right_below_max_vertex = max_rectangle[len(max_rectangle) - 2]
        left_above_min_vertex = min_rectangle[0]
        left_below_min_vertex = min_rectangle[1]
        cluster_values.append(
            [left_above_min_vertex, left_below_min_vertex, right_above_max_vertex, right_below_max_vertex])


    img = np.array(image[:,:,::-1])
    for i, box in enumerate(cluster_values):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        # print(poly)
        rect = cv2.boundingRect(poly)
        x,y,w,h = rect
        if h/w < 0.5:
    #     croped1 = img[y-5:y+h+5, x-5:x+w+5].copy()
            croped1 = img[abs(abs(y)-10):abs(y)+h+10, abs(abs(x)-10):abs(x)+w+10].copy()
            cv2.imwrite("./QD_image/page_segments/%s.png"%image_index, croped1)
            image_index += 1

    poly = False
    refine = False
    show_time = False
    refine_net = None
