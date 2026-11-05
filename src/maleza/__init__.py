import os
import sys

# Variables globales de otras instalaciones GIS (ej. MapWinGIS) hacen que rasterio/pyproj busquen
# proj.db / gdal-data viejos. Las wheels traen sus propios datos: ignoramos las que apunten afuera del venv.
for _var in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA", "GDAL_DRIVER_PATH"):
    _val = os.environ.get(_var)
    if _val and not os.path.abspath(_val).startswith(os.path.abspath(sys.prefix)):
        del os.environ[_var]
