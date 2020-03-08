from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import gpx_parser as parser
import os
import sys
import argparse
from math import radians, cos, sin, asin, sqrt
import cv2
import aux

folders, map_aspect_ratio, limit, num_bins, bbox, invert, colormap = aux.get_argparser_values()

latitudes, longitudes = aux.load_latlon(folders, bbox)

#pyplot setup
fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis('off')
ax.set_aspect(map_aspect_ratio)

#histogram creation
H, xedges, yedges = np.histogram2d(longitudes, latitudes, bins=num_bins, density=False)

#slight blur against noise
H = cv2.GaussianBlur(H, (5,5), sigmaX=0.5, sigmaY=0.5)

#truncation of ridiculously large values
H[H >= limit] = limit
image=plt.imshow(np.flip(H.T, 0), cmap=colormap, aspect=map_aspect_ratio, interpolation='bilinear', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
aux.savefig(folders, limit, 'DENSITYMAP', invert)

