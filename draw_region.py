#!/usr/bin/env python
# encoding: utf-8

# import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

overlay_file = './Data/cancer/case0001/C_0001_1.RIGHT_CC.OVERLAY'

img_file = './Data/cancer/case0001/C_0001_1.RIGHT_CC.jpg'
# LINE_COLOR = [0, 1, 0]
LINE_COLOR = 0

DIRECTIONS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1,
                                                                           -1)]


def nextpos(oldpos, direction):
    newpos = []
    delta = DIRECTIONS[direction]
    newpos.append(oldpos[0] + delta[0])
    newpos.append(oldpos[1] + delta[1])
    return newpos


def crop(boundary_points, im, x_pad, y_pad):
    # find the max and min of points
    ymin, xmin = im.shape
    ymax, xmax = -1, -1
    for point in boundary_points:
        if xmin > point[0]:
            xmin = point[0]
        if ymin > point[1]:
            ymin = point[1]
        if xmax < point[0]:
            xmax = point[0]
        if ymax < point[1]:
            ymax = point[1]

    xmin -= x_pad
    ymin -= y_pad
    xmin = 0 if xmin < 0 else xmin
    xmax += x_pad
    ymax += y_pad
    xmax = im.shape[1] if xmax >= im.shape[1] else xmax
    ymax = im.shape[0] if ymax >= im.shape[0] else ymax

    return im[ymin:ymax + 1, xmin:xmax + 1]


def generate_points(start_pos, directions):
    points = []
    cur_pos = start_pos
    points.append(cur_pos)

    for direction in directions:
        cur_pos = nextpos(cur_pos, direction)
        points.append(cur_pos)
    return np.asarray(points, np.int32)


def test_directions():
    # start from any pos, and walk aroud,finally it arrives the original pos
    start_pos = (5, 5)
    cur_pos = start_pos
    course = [0, 2, 4, 4, 6, 6, 0, 0]
    for direction in course:
        cur_pos = nextpos(cur_pos, direction)
    if tuple(cur_pos) == start_pos:
        print("SUCCESS")
    else:
        print("ERROR")
        print(cur_pos, start_pos)

    # blank image
    blank_im = np.zeros((10, 10), dtype='uint8')
    blank_im[...] = 255

    drawoutline(start_pos, course, blank_im)

    print(blank_im)
    imshow(blank_im)


def imshow(im, cmap='gray'):
    plt.imshow(im, cmap=cmap, vmin=0, vmax=255)
    plt.axis('off')
    plt.show()


def checkadjuct(pos1, pos2):
    if abs(pos1[0] - pos2[0]) < 2 and abs(pos1[1] - pos2[1]) < 2:
        return True
    return False


def checklegal(pos, im_shape):
    if pos[1]>=0 and pos[0]<im_shape[0]\
            and pos[0]>=0 and pos[1]<im_shape[1]:
        return True
    return False


def drawoutline(start_pos, boundary, im):
    cur_pos = start_pos
    boundary_points = []
    for idx, direction in enumerate(boundary):
        boundary_points.append(cur_pos)
        im[cur_pos[1], cur_pos[0], ...] = LINE_COLOR
        tmp_nextpos = nextpos(cur_pos, direction)
        assert checkadjuct(tmp_nextpos, cur_pos)
        assert checklegal(
            tmp_nextpos,
            im.shape), 'tmp_nextpos: ' + str(tmp_nextpos) + str(idx)
        cur_pos = tmp_nextpos
    im[cur_pos[1], cur_pos[0], ...] = LINE_COLOR
    print('start_pos: ', start_pos)
    print('cur_pos: ', cur_pos)
    if not tuple(start_pos) == tuple(cur_pos):
        print("warning: curve is not closed")
    return boundary_points


def fillinner(boundary_points, im):
    # select any inner point randomly
    start_point = []
    start_point.append(boundary_points[0][0] - 1)
    start_point.append(boundary_points[1][0] - 1)
    boundary_points = [tuple(point) for point in boundary_points]
    boundary_points = set(boundary_points)
    points = [start_point]
    while True:
        if len(points) == 0:
            break
        sys.stdout.write('\rnum_points: {}/{}'.format(len(points), im.size))
        sys.stdout.flush()
        point = points[0]
        # draw black in the point
        im[point[1], point[0]] = LINE_COLOR
        for direction in range(len(DIRECTIONS)):
            next_point = nextpos(point, direction)

            if not checklegal(next_point, im.shape):
                continue

            if tuple(point
                     ) in boundary_points or im[next_point[1],
                                                next_point[0]] == LINE_COLOR:
                continue
            points.append(next_point)

        del points[0]


def test_crop():
    pad = 100
    im = cv2.imread(img_file)
    boundary_points = generate_points()
    crop(boundary_points, im, pad, pad)


def main():
    pad = 100
    # read image
    im = plt.imread(img_file)
    im = np.array(im, dtype='uint8')

    # debug
    # imshow(im)

    # blank image
    # blank_im = np.zeros_like(im)
    # blank_im = np.zeros((800, 800))
    # blank_im[...] = 100
    # print(blank_im.shape)
    print(im.shape)

    # read overlay
    with open(overlay_file) as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            if not line[0].isdigit():
                continue
            # convert str to int
            line = [int(item) for item in line[:-1]]

            # the first two values of each chain code are the starting column and row of the chain code
            start_x, start_y = line[:2]
            boundary = line[2:]
            # boundary_points = drawoutline((start_x, start_y), boundary, im)
            # fillinner(boundary_points, im)
            boundary_points = generate_points((start_x, start_y), boundary)
            cv2.fillConvexPoly(im, boundary_points, 1)
            im = crop(boundary_points, im, pad, pad)

    # display after drawing
    # imshow(im)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()
    plt.imsave('blocked.jpg', im, cmap='gray', vmin=0, vmax=255)

    # drawoutline((500, 500), boundary[:300], blank_im)
    # imshow(blank_im)


main()
# test_directions()
