import numpy as np
import matplotlib.pyplot as plt
import csv
import gpx_parser as parser
import os
import sys
import argparse
import cv2


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


argparser = argparse.ArgumentParser()
argparser.add_argument('-l', '--limit', help='sets the limit in each bin, 30 by default')
argparser.add_argument('--folder', required=True, help='folder path containing *.GPX files')
limit = 30
limit = int(argparser.parse_args().limit)
fold_name = argparser.parse_args().folder


latitudes = []
longitudes = []


dir_name = fold_name + '/'


for file in os.listdir(dir_name):
    file = dir_name + file
    if file.endswith(".GPX"):
        with open(file, 'r') as gpx_file:
            gpx = parser.parse(gpx_file)
            track = gpx[0]

            for point in track.points:
                if point.latitude > 51.518:
                    continue
                latitudes.append(point.latitude)
                longitudes.append(point.longitude)

latitudes = np.array(latitudes)
longitudes = np.array(longitudes)

x_min = np.min(longitudes)
x_max = np.max(longitudes)
y_min = np.min(latitudes)
y_max = np.max(latitudes)

asp_ratio = (y_max-y_min) / (x_max-x_min)
bins = 2000


matrix = np.zeros((int(7.4/4.6*asp_ratio*bins), bins), dtype=float)
print(matrix.shape)
for i in range(0, latitudes.shape[0]-1):
    y1 = int((latitudes[i]-y_min)/(y_max-y_min)*matrix.shape[0])-1
    x1 = int((longitudes[i]-x_min)/(x_max-x_min)*matrix.shape[1])-1
    y2 = int((latitudes[i+1]-y_min)/(y_max-y_min)*matrix.shape[0])-1
    x2 = int((longitudes[i+1]-x_min)/(x_max-x_min)*matrix.shape[1])-1

    if (y1-y2)*(y1-y2)+(x1-x2)*(x1-x2) > 2000:
        continue

    line = drawline(x1,y1, x2,y2)
    line.append([y1, x1])
    line.append([y2, x2])
    line = np.array(line)
    if line.shape[0] == 0:
        line = np.array([[y1, x1]])

    matrix[line[:,0], line[:,1]] += 1

print(np.max(matrix))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis('off')

#ax.set_aspect(7.4/4.6)
#H, xedges, yedges = np.histogram2d(longitudes, latitudes, bins=1200, density=False)

matrix = cv2.GaussianBlur(matrix, (5,5), sigmaX=0.5, sigmaY=0.5)
matrix[matrix >= limit] = limit
plt.imshow(np.flip(matrix,0), cmap='inferno', aspect='equal', interpolation='bilinear')

plt.savefig('line'+fold_name+'_lim'+str(limit)+'.png', dpi=2000, bbox_inches='tight', pad_inches=0, transparent=True)
