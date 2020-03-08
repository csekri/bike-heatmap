import mapnik
import numpy as np
import cv2
import aux
import matplotlib.pyplot as plt


map = mapnik.Map(8000,8000)
#map.zoom_to_box(mapnik.Box2d(-2.7309325500000003,51.3952918,-2.4723757499999994,51.567663))

mapnik.load_map(map, 'style.xml')
map.aspect_fix_mode = mapnik.aspect_fix_mode.ADJUST_CANVAS_HEIGHT

#map.zoom_all()
map.zoom_to_box(mapnik.Box2d(-303902.09854614607,6692390.569193574,-279382.1995870749,6717812.607185805))

p = mapnik.Projection("+init=epsg:3857")
bbox = p.inverse(map.envelope())

mapnik.render_to_file(map,'world.png', 'png')
print ('rendered image to "world.png"')

folders, map_aspect_ratio, limit, num_bins, bbox, invert, colormap = aux.get_argparser_values()

latitudes, longitudes = aux.load_latlon(folders, bbox)

bbox = p.inverse(map.envelope())
x_min, x_max, y_min, y_max, aspect_ratio = aux.get_image_aspect_ratio(latitudes, longitudes)

x_percent = (x_max-x_min)/(bbox.maxx-bbox.minx)
y_percent = (y_max-y_min)/(bbox.maxy-bbox.miny)

WIDTH = int(map.width*x_percent)
HEIGHT = int(map.height*y_percent)

#pyplot setup
fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis('off')
ax.set_aspect(map_aspect_ratio)

#histogram creation
H, xedges, yedges = np.histogram2d(longitudes, latitudes, bins=num_bins, density=False)
H[H >= limit] = limit
plt.imshow(np.flip(H.T, 0), cmap=colormap, aspect=map_aspect_ratio, interpolation='bilinear', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.savefig('h.png', dpi=1200, bbox_inches='tight', pad_inches=0, transparent=True)

#slight blur against noise
#H = cv2.GaussianBlur(H, (5,5), sigmaX=0.5, sigmaY=0.5)
H = cv2.imread('h.png')
H = H.max()-H
H= cv2.resize(H, (WIDTH, HEIGHT))

img = cv2.imread('world.png')

for y in range(HEIGHT):
    for x in range(WIDTH):
        if H[y,x][0] < 250 and H[y,x][1] < 250 and H[y,x][2] < 250:
            img[y + int(map.height*(bbox.maxy-y_max)/(bbox.maxy-bbox.miny)), x + int(map.width*(x_min-bbox.minx)/(bbox.maxx-bbox.minx)),:] = H[y,x,:]
H[10,:] = 0
cv2.imwrite('out.jpg', img)
#truncation of ridiculously large values
