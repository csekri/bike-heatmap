import numpy as np
import matplotlib.pyplot as plt
import csv
import gpx_parser as parser
import os
import sys
import argparse
import cv2
import aux


def drawline(x0, y0, x1, y1):
    dx = abs(x1 - x0)    # distance to travel in X
    dy = abs(y1 - y0)    # distance to travel in Y
    pixels = []

    ix = 1 if x0 < x1 else -1
    iy = 1 if y0 < y1 else -1
    e = 0                # Current error

    for i in range(dx + dy):
        pixels.append([y0, x0])
        e1 = e + dy
        e2 = e - dx
        if abs(e1) < abs(e2):
            # Error will be smaller moving on X
            x0 += ix
            e = e1
        else:
            # Error will be smaller moving on Y
            y0 += iy
            e = e2
    return pixels

folders, map_aspect_ratio, limit, num_bins, bbox, invert, colormap = aux.get_argparser_values()

latitudes, longitudes = aux.load_latlon(folders, bbox)

x_min, x_max, y_min, y_max, aspect_ratio = aux.get_image_aspect_ratio(latitudes, longitudes)


matrix = np.zeros((int(aspect_ratio*map_aspect_ratio*num_bins), num_bins), dtype=float)

for i in range(0, latitudes.shape[0]-1):
    y1 = int((latitudes[i]-y_min)/(y_max-y_min)*matrix.shape[0])-1
    x1 = int((longitudes[i]-x_min)/(x_max-x_min)*matrix.shape[1])-1
    y2 = int((latitudes[i+1]-y_min)/(y_max-y_min)*matrix.shape[0])-1
    x2 = int((longitudes[i+1]-x_min)/(x_max-x_min)*matrix.shape[1])-1

    if (y1-y2)*(y1-y2)+(x1-x2)*(x1-x2) > 100:
        continue

    line = drawline(x1,y1, x2,y2)
    line.append([y1, x1])
    line.append([y2, x2])
    line = np.array(line)
    if line.shape[0] == 0:
        line = np.array([[y1, x1]])

    matrix[line[:,0], line[:,1]] += 1

fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis('off')

aux.imshow(matrix, limit, colormap)

aux.savefig(folders, limit, 'POLYLINEMAP', invert)

