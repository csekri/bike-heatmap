from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import gpx_parser as parser
import os
import sys
import argparse
from math import radians, cos, sin, asin, sqrt
import cv2

def get_argparser_values():
    #predefined values
    limit = 30
    num_bins = 1200
    bristol_aspect_ratio = 7.4/4.6
    aspect_ratio = bristol_aspect_ratio
    colormap = 'inferno'
    isinvert = False

    bristol_city_bbox = [51.429, 51.518, -2.630, -2.540]
    bristol_vicinity_bbox = [51.183, 51.929, -3.797, -1.787]
    europe_bbox = [40.4, 55.3, -4.6, 25]
    balint_bbox = [53.412, 53.509, -2.311, -2.128]
    balint_uk_bbox = [51.122, 55.072, -4.930, 0.956]
    bbox = []

    #arguments we can/must pass
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-l', '--limit', help='sets the limit in each bin, 30 by default')
    argparser.add_argument('-f', '--folder', required=True, help='folder path containing *.GPX files')
    argparser.add_argument('-b', '--bins', help='sets the number of bins for each dimensions of the image in the histogram')
    argparser.add_argument('--aspect-ratio', help='sets the aspect ratio of the output image')
    argparser.add_argument('--bbox', help='bounding box of the area we want')
    argparser.add_argument('-i', '--invert', help='invert colors "true" or "false"')
    argparser.add_argument('-c', '--colormap', help='name of a matplotlib colormap')

    #values updated as defined in command line
    fold_name = argparser.parse_args().folder
    if argparser.parse_args().aspect_ratio != None:
        aspect_ratio = int(argparser.parse_args().aspect_ratio)
    if argparser.parse_args().limit != None:
        limit = int(argparser.parse_args().limit)
    if argparser.parse_args().bins != None:
        num_bins = int(argparser.parse_args().bins)
    if argparser.parse_args().bbox != None:
        bbox = argparser.parse_args().bbox
        if bbox == 'europe':
            bbox = europe_bbox
        elif bbox == 'bristol_vicinity':
            bbox = bristol_vicinity_bbox
        elif bbox == 'bristol_city':
            bbox = bristol_city_bbox
        elif bbox == 'balint':
            bbox = balint_bbox
        elif bbox == 'balint_uk':
            bbox = balint_uk_bbox
    if argparser.parse_args().invert != None:
        isinvert = argparser.parse_args().invert
        if isinvert == 'true':
            isinvert = True
        elif isinvert == 'false':
            isinvert = False
        else:
            print('Invert argument is invalid.')
            sys.exit(0)
    if argparser.parse_args().colormap != None:
        colormap = argparser.parse_args().colormap


    #we use this because we want to also have an all folders compile as well, so folder names are stored in a list
    folders = []
    if fold_name == 'all':
        folders = next(os.walk('.'))[1]
    else:
        folders = [fold_name]
    folders = list(map(lambda x: x+'/', folders))

    return folders, aspect_ratio, limit, num_bins, bbox, isinvert, colormap


def load_latlon(folders, bbox):
    #loop first in folders, then inside a particular folder then in a particular file
    latitudes = []
    longitudes = []
    for dir_name in folders:
        for file in os.listdir(dir_name):
            file = dir_name + file
            if file.endswith(".GPX"):
                with open(file, 'r') as gpx_file:
                    gpx = parser.parse(gpx_file)
                    track = gpx[0]

                    for point in track.points:
                        if bbox == [] or (bbox[0]<point.latitude<bbox[1] and bbox[2]<point.longitude<bbox[3]):
                            latitudes.append(point.latitude)
                            longitudes.append(point.longitude)
    return np.array(latitudes), np.array(longitudes)

def get_image_aspect_ratio(latitudes, longitudes):
    x_min = np.min(longitudes)
    x_max = np.max(longitudes)
    y_min = np.min(latitudes)
    y_max = np.max(latitudes)
    return x_min, x_max, y_min, y_max, (y_max-y_min) / (x_max-x_min)

def imshow(matrix, limit, colormap):
    #matrix = cv2.GaussianBlur(matrix, (5,5), sigmaX=0.5, sigmaY=0.5)
    matrix[matrix >= limit] = limit
    return plt.imshow(np.flip(matrix,0), cmap=colormap, aspect='equal', interpolation='bilinear')

def savefig(folders, limit, extra, isinvert):
    container_folder = 'map_images'
    if not os.path.exists(container_folder):
        os.makedirs(container_folder)
    container_folder += '/'
    if len(folders) > 1:
        fname = str(extra)+': all folders lim='+str(limit)+'.png'
    else:
        fname = str(extra)+' '+folders[0][0:-1]+' lim='+str(limit)+'.png'
    plt.savefig(container_folder+fname, dpi=1400, bbox_inches='tight', pad_inches=0, transparent=True)
    if isinvert == True:
        to_be_inverted_and_deleted = cv2.imread(container_folder+fname)
        to_be_inverted_and_deleted = to_be_inverted_and_deleted.max()-to_be_inverted_and_deleted
        cv2.imwrite(container_folder+'INV'+fname[0:-4]+'.jpg', to_be_inverted_and_deleted)

        os.remove(container_folder+fname)
