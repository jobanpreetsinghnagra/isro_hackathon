# Quick example to get you started
from pathlib import Path
import subprocess
import geopandas as gpd

# Your PBF file path
pbf_file = "data/your_file.osm.pbf"

# Extract roads (most useful for spatial analysis)
subprocess.run(f'osmium tags-filter {pbf_file} w/highway -o data/roads.osm.pbf', shell=True)

# Convert to shapefile
subprocess.run(f'ogr2ogr -f "ESRI Shapefile" data/roads.shp data/roads.osm.pbf lines', shell=True)

# Load with GeoPandas
roads = gpd.read_file("data/roads.shp")
print(f"Loaded {len(roads)} roads")
print(roads.head())