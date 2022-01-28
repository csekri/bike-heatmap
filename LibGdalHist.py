from __future__ import print_function, annotations
import numpy as np
from matplotlib import cm # colormap
import os
import glob
import sys
import argparse
from cv2 import GaussianBlur # faster than scipy convolution
from fiona import open as fiona_open # turn .GPX into a list of coordinates
from osgeo import gdal
from osgeo import osr
from numba import jit
from typing import List, Tuple

INVALID_COORDINATE = 1e5



@jit(nopython=True)
def faster_draw_hist(
        H: np.ndarray,
        bins: int,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        area: np.ndarray,
        inverse: bool=False):
    """Convert a list of coordinates inside a bounding area to a matrix."""
    xmin, ymin, xmax, ymax = area[0], area[1], area[2], area[3]

    for lat, lon in zip(latitudes, longitudes):
        x = int((lon-xmin) / (xmax - xmin) * bins)
        y = int((lat-ymin) / (ymax - ymin) * bins)
        if 1 < x < bins-1 and 1 < y < bins-1:
            H[y,x] += 4.0

            H[y+1,x] += 2.0
            H[y-1,x] += 2.0
            H[y,x+1] += 2.0
            H[y,x-1] += 2.0

            H[y+1,x+1] += 1.0
            H[y-1,x-1] += 1.0
            H[y-1,x+1] += 1.0
            H[y+1,x-1] += 1.0

@jit(nopython=True)
def fill_line(
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        area: np.ndarray,
        margin: float,
        num_bins: int,
        emph_stops: int) -> Tuple[List[float], List[float]]:
    longs = []
    lats = []
    x_min, y_min, x_max, y_max = area[0], area[1], area[2], area[3]
    bins = num_bins*1

    for i in range(longitudes.shape[0]-1):
        num_x = (longitudes[i+1]-longitudes[i])*bins/(x_max-x_min)
        num_y = (latitudes[i+1]-latitudes[i])*bins/(y_max-y_min)
        num = int((num_x*num_x+num_y*num_y)**0.5)

        if num > 40:
            continue
        else:
            num = 20
            dx = (longitudes[i+1]-longitudes[i]) / num
            dy = (latitudes[i+1]-latitudes[i]) / num

            x = longitudes[i]
            y = latitudes[i]
            for j in range(num):
                x += dx
                y += dy
                longs.append(x)
                lats.append(y)
        for k in range(emph_stops-1):
            longs.append(longitudes[i])
            lats.append(latitudes[i])
    return longs, lats


class LibGdalHist:
    def __init__(self):
        self.limit = 30
        self.num_bins = 1200
        self.area = None
        self.add_min = 0
        self.cmap = 'inferno'
        self.folder_name = ''
        self.filenames = []
        self.latitudes = np.array([])
        self.longitudes = np.array([])
        self.margin = 0.08
        self.smoothing = 7
        self.emphasise_stops = 1
        self.get_argparser_values()

        self.load_latlon(True)


    def get_argparser_values(self: LibGdalHist):
        argparser = argparse.ArgumentParser()
        argparser.add_argument('-l', '--limit', default=30, help='sets the limit in each bin, 30 by default')
        argparser.add_argument('-f', '--folder', required=True, help='folder path containing *.GPX files')
        argparser.add_argument('-b', '--bins', default=1200, help='sets the number of bins for each dimensions of the image in the histogram')
        argparser.add_argument('-a', '--area', default=',,,', help='bounding box of the area we want (long_min,lat_min,long_max,lat_max)')
        argparser.add_argument('-c', '--colormap', default='inferno', help='name of a matplotlib colormap')
        argparser.add_argument('-m', '--add_min', default=0, help='takes out lower value colours in the colormap')
        argparser.add_argument("-s", '--smoothing', default=7, help='adds gaussian smoothing')
        argparser.add_argument("-e", '--emph_stops', default=1, help='emphasise stops by this factor')

        parsed = argparser.parse_args()

        self.folder_name = parsed.folder
        self.filenames =  glob.iglob(self.folder_name + '/' + '**/*.[gG][pP][xX]', recursive=True)
        assert bool(self.filenames)

        self.limit = int(parsed.limit)
        self.num_bins = int(parsed.bins)
        self.add_min = int(parsed.add_min)
        self.smoothing = int(parsed.smoothing)
        self.cmap = parsed.colormap
        self.emphasise_stops = int(parsed.emph_stops)

        self.area = argparser.parse_args().area
        self.area = self.area.split(',')
        for i in range(len(self.area)):
            if self.area[i] == '':
                self.area[i] = str(INVALID_COORDINATE)
        self.area = list(map(float, self.area))

        assert len(self.area) == 4
        assert self.num_bins > 50
        assert self.limit > 0
        assert self.emphasise_stops > 0



    def save_geotiff(
            self: LibGdalHist,
            fname: str,
            image: np.ndarray,
            alpha: np.ndarray,
            rgb: str='321'):
            def get_geotransform(
                    area: np.ndarray,
                    H: int,
                    W: int) -> Tuple[float, float, float, float, float]:
                xmin, ymin, xmax, ymax = area
                xres = (xmax - xmin) / H
                yres = (ymax - ymin) / W
                return (xmin, xres, 0.0, ymax, 0.0, -yres)

            if image.ndim == 2:
                channels = 1
            elif image.ndim == 3:
                channels = image.shape[2]
            else:
                print('Error encountered in the number of dimensions of an image.')

            W,H = (image.shape[0], image.shape[1])

            # create the 3 or 4-band raster file
            if channels == 3 or channels == 1:
                dst_ds = gdal.GetDriverByName('GTiff').Create(fname, H, W, 3, gdal.GDT_Byte)
            elif channels == 4:
                dst_ds = gdal.GetDriverByName('GTiff').Create(fname, H, W, 4, gdal.GDT_Byte)

            actual_area = self.area

            dst_ds.SetGeoTransform(get_geotransform(actual_area, H, W))    # specify coords
            srs = osr.SpatialReference()            # establish encoding
            srs.ImportFromEPSG(4326)                # WGS84 lat/long
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file

            if channels == 1:
                dst_ds.GetRasterBand(1).WriteArray(image)   # write r-band to the raster
                dst_ds.GetRasterBand(2).WriteArray(image)   # write g-band to the raster
                dst_ds.GetRasterBand(3).WriteArray(image)   # write b-band to the raster
            elif channels == 3 or channels == 4:
                r, g, b = (int(rgb[0])-1, int(rgb[1])-1, int(rgb[2])-1)
                dst_ds.GetRasterBand(1).WriteArray(image[:,:,r])   # write r-band to the raster
                dst_ds.GetRasterBand(2).WriteArray(image[:,:,g])   # write g-band to the raster
                dst_ds.GetRasterBand(3).WriteArray(image[:,:,b])   # write b-band to the raster

                dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
                dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
                dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
            if channels == 4:
                dst_ds.GetRasterBand(4).WriteArray(alpha)   # write b-band to the raster
                dst_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

            dst_ds.FlushCache()                     # write to disk
            dst_ds = None


    # mode can be 'point' or 'line'
    def load_latlon(self: LibGdalHist, line_fill: bool=False):
        # loop first in folders, then inside a particular folder then in a particular file
        for filename in self.filenames:
            print('processing: ' + filename)
            points = np.array(fiona_open(filename, layer='tracks')[0]['geometry']['coordinates'])
            self.longitudes = np.hstack((self.longitudes, points[0, :, 0].flatten()))
            self.latitudes = np.hstack((self.latitudes, points[0, :, 1].flatten()))

        if self.area[0] == INVALID_COORDINATE:
            self.area[0] = self.longitudes.min()
        if self.area[1] == INVALID_COORDINATE:
            self.area[1] = self.latitudes.min()
        if self.area[2] == INVALID_COORDINATE:
            self.area[2] = self.longitudes.max()
        if self.area[3] == INVALID_COORDINATE:
            self.area[3] = self.latitudes.max()
        xmin, ymin, xmax, ymax = self.area
        dx = xmax - xmin
        dy = ymax - ymin
        xmin_ = xmin - self.margin * dx
        xmax_ = xmax + self.margin * dx
        ymin_ = ymin - self.margin * dy
        ymax_ = ymax + self.margin * dy
        self.area = np.array([xmin_, ymin_, xmax_, ymax_])

        if line_fill:
            longs, lats = fill_line(self.longitudes, self.latitudes, self.area, self.margin, self.num_bins, self.emphasise_stops)
            self.longitudes = np.array(longs)
            self.latitudes = np.array(lats)



    def __str__(self: LibGdalHist):
        return "MAP_F-%s_L%s_B%s_M%s" % (self.folder_name.split("/")[0], self.limit, self.num_bins, self.add_min)


    def alpha(self: LibGdalHist, image: np.ndarray) -> np.ndarray:
        mini = image.min()
        maxi = image.max()
        image = (image - mini) / (maxi-mini)
        #return (np.sin(avg*(np.pi/2))**2) * 255
        return image ** 0.4 * 255


    def draw_hist(self, tiff_name='myGeoTIFF.tif', inverse=False):
        H = np.zeros((self.num_bins, self.num_bins), dtype=np.float)
        faster_draw_hist(H, self.num_bins, self.latitudes, self.longitudes, self.area, inverse)

        H = np.flip(np.flip(H), axis=1)
        H[H >= self.limit] = self.limit

        H = H + self.add_min #np.where(H>0, H+self.add_min, H)
        if self.smoothing > 0:
            #H = np.array(Image.fromarray(H).resize(size=(H.shape[0]*2, H.shape[1]*2)), dtype=float)
            H = GaussianBlur(H, (2*self.smoothing+1, 2*self.smoothing+1), 1.5)
            #H = np.array(Image.fromarray(H).resize(size=(H.shape[0]*1//2, H.shape[1]*1//2)), dtype=float)
        my_cm = cm.get_cmap(self.cmap)
        H = H / np.max(H)
        alpha = self.alpha(H)
        H = 255 * my_cm(H)

        if inverse:
            self.save_geotiff(tiff_name, 255-H, alpha, rgb='123')
        else:
            self.save_geotiff(tiff_name, H, alpha, rgb='123')


#
