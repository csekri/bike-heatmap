import numpy as np
import matplotlib.pyplot as plt
import csv
import gpx_parser as parser
import os
import sys
import argparse



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
bins = 1500


matrix = np.zeros((int(7.4/4.6*asp_ratio*bins), bins), dtype=int)
print(matrix.shape)
for i in range(0, latitudes.shape[0]-1):
    y = int((latitudes[i]-y_min)/(y_max-y_min)*matrix.shape[0])-1
    x = int((longitudes[i]-x_min)/(x_max-x_min)*matrix.shape[1])-1

    #ball = np.array([[y, x],[y+1,x],[y-1,x],[y,x+1],[y+1,x+1],[y-1,x+1],[y,x-1],[y+1,x-1],[y-1,x-1]])
    ball = np.array([[y, x],[y+1,x],[y,x+1],[y+1,x+1]])

    try:
        matrix[ball[:,0], ball[:,1]] += 1
    except IndexError:
        pass

print(np.max(matrix))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis('off')

#ax.set_aspect(7.4/4.6)
#H, xedges, yedges = np.histogram2d(longitudes, latitudes, bins=1200, density=False)

limit = 40
matrix[matrix >= limit] = limit
plt.imshow(np.flip(matrix,0), cmap='inferno', aspect='equal', interpolation='bilinear')

plt.savefig('ball'+fold_name+'_lim'+str(limit)+'.png', dpi=2000, bbox_inches='tight', pad_inches=0, transparent=True)
