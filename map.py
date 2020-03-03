from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import gpx_parser as parser
import os
import sys
import argparse
from math import radians, cos, sin, asin, sqrt
import cv2

#predefined values
limit = 30
num_bins = 1200
bristol_aspect_ratio = 7.4/4.6
aspect_ratio = bristol_aspect_ratio



#arguments we can/must pass
argparser = argparse.ArgumentParser()
argparser.add_argument('-l', '--limit', help='sets the limit in each bin, 30 by default')
argparser.add_argument('-f', '--folder', required=True, help='folder path containing *.GPX files')
argparser.add_argument('-b', '--bins', help='sets the number of bins for each dimensions of the image in the histogram')
argparser.add_argument('--aspect-ratio', help='sets the aspect ratio of the output image')

#values updated as defined in command line
fold_name = argparser.parse_args().folder
if argparser.parse_args().aspect_ratio != None:
    aspect_ratio = int(argparser.parse_args().aspect_ratio)
if argparser.parse_args().limit != None:
    limit = int(argparser.parse_args().limit)
if argparser.parse_args().bins != None:
    num_bins = int(argparser.parse_args().bins)


#we collect ALL latitude and longitude values in these lists
latitudes = []
longitudes = []

#we use this because we want to also have an all folders compile as well, so folder names are stored in a list
folders = []
if fold_name == 'all':
    folders = next(os.walk('.'))[1]
else:
    folders = [fold_name]
folders = list(map(lambda x: x+'/', folders))


#loop first in folders, then inside a particular folder then in a particular file
for dir_name in folders:
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

#numpy conversion
latitudes = np.array(latitudes)
longitudes = np.array(longitudes)

#pyplot setup
fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis('off')
ax.set_aspect(aspect_ratio)

#histogram creation
H, xedges, yedges = np.histogram2d(longitudes, latitudes, bins=num_bins, density=False)

#slight blur against noise
H = cv2.GaussianBlur(H, (5,5), sigmaX=0.5, sigmaY=0.5)

#truncation of ridiculously large values
H[H >= limit] = limit

plt.imshow(np.flip(H.T, 0), cmap='inferno', aspect=aspect_ratio, interpolation='bilinear', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.savefig('dens'+fold_name+'_lim'+str(limit)+'.png', dpi=1400, bbox_inches='tight', pad_inches=0, transparent=True)
