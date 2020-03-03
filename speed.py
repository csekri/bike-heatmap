import numpy as np
import matplotlib.pyplot as plt
import csv
import gpx_parser as parser
import os
import sys
import argparse
from math import radians, cos, sin, asin, sqrt
import datetime
import cv2

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371*1000 # Radius of earth in kilometers. Use 3956 for miles
    return c*r

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
times = []
speed = []

dir_name = fold_name + '/'


for file in os.listdir(dir_name):
    file = dir_name + file
    if file.endswith(".GPX"):
        with open(file, 'r') as gpx_file:
            gpx = parser.parse(gpx_file)
            track = gpx[0]

            for i, point in enumerate(track.points):
                if point.latitude > 51.518:
                    continue
                latitudes.append(point.latitude)
                longitudes.append(point.longitude)
                times.append((point.time - datetime.datetime(2000,1,1)).total_seconds())


for i in range(2,len(latitudes)):
    speed_value = haversine(longitudes[i-2], latitudes[i-2], longitudes[i], latitudes[i]) /  (times[i]-times[i-2])
    speed.append((latitudes[i-1], longitudes[i-1], speed_value))



latitudes = np.array(latitudes)
longitudes = np.array(longitudes)


x_min = np.min(longitudes)
x_max = np.max(longitudes)
y_min = np.min(latitudes)
y_max = np.max(latitudes)

asp_ratio = (y_max-y_min) / (x_max-x_min)
bins = 1500

matrix = np.zeros((int(7.4/4.6*asp_ratio*bins), bins), dtype=float)
counter = np.zeros((int(7.4/4.6*asp_ratio*bins), bins), dtype=float)
print(matrix.shape)
for i in range(0, len(speed)-1):
    y1 = int((latitudes[i]-y_min)/(y_max-y_min)*matrix.shape[0])-1
    x1 = int((longitudes[i]-x_min)/(x_max-x_min)*matrix.shape[1])-1
    y2 = int((latitudes[i+1]-y_min)/(y_max-y_min)*matrix.shape[0])-1
    x2 = int((longitudes[i+1]-x_min)/(x_max-x_min)*matrix.shape[1])-1

    #ball = np.array([[y, x],[y+1,x],[y-1,x],[y,x+1],[y+1,x+1],[y-1,x+1],[y,x-1],[y+1,x-1],[y-1,x-1]])
    #ball = np.array([[y, x],[y+1,x],[y,x+1],[y+1,x+1]])

    if (y1-y2)*(y1-y2)+(x1-x2)*(x1-x2) > 1000:
        continue

    line = drawline(x1,y1, x2,y2)
    line.append([y1, x1])
    line.append([y2, x2])
    line = np.array(line)
    if line.shape[0] == 0:
        line = np.array([[y1, x1]])

    if 2 < x1 < matrix.shape[1]-2 and 2 < y1 < matrix.shape[0]-2:
        if 2 < x2 < matrix.shape[1]-2 and 2 < y2 < matrix.shape[0]-2:
            matrix[line[:,0],line[:,1]] = (matrix[line[:,0],line[:,1]]*counter[line[:,0],line[:,1]]+speed[i][2])/(counter[line[:,0],line[:,1]]+1)
            counter[line[:,0],line[:,1]] += 1


matrix = np.flip(matrix, 0)

print('Max speed=' + str(np.max(matrix)))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis('off')

#ax.set_aspect(7.4/4.6)
#H, xedges, yedges = np.histogram2d(longitudes, latitudes, bins=1200, density=False)

#cv2.imwrite('file.jpg', matrix/np.max(matrix)*255)
matrix[matrix >= limit] = limit
plt.imshow(matrix, cmap='hot', aspect='equal', interpolation='bilinear')
plt.colorbar()
plt.savefig('speed'+fold_name+'_lim'+str(limit)+'.png', dpi=1000, bbox_inches='tight', pad_inches=0, transparent=True)
