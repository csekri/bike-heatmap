import mapnik
import numpy as np
from LibGdalHist import LibGdalHist
import matplotlib.pyplot as plt

lgh = LibGdalHist()
lgh.draw_hist(inverse=False)

# The resolution of the output image can be increased with the parameter below
resolution = 500
map = mapnik.Map(resolution, resolution)
mapnik.load_map(map, 'buildings.xml')
area = mapnik.Box2d(*lgh.area)
p = mapnik.Projection("+init=epsg:3857")
map.aspect_fix_mode = mapnik.aspect_fix_mode.ADJUST_CANVAS_HEIGHT
map.zoom_to_box(p.forward(area))
# map.zoom_all()
# bbox = p.forward(map.envelope())
mapnik.render_to_file(map, str(lgh) + '.png', 'png')
print ('rendered image to "world.png"')
