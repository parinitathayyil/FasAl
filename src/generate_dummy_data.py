import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_dummy_geotiff(filename, width, height, num_bands=3, pixel_size=10.0):
    folder = os.path.dirname(filename)
    
    # If a file exists where the directory should be, remove it first
    if os.path.isfile(folder):
        os.remove(folder)
        
    os.makedirs(folder, exist_ok=True)
    
    # Generate synthetic spectral values
    data = np.random.uniform(0.1, 0.9, (num_bands, height, width)).astype(np.float32)
    
    # Spatial transform
    transform = from_origin(-120.5, 37.5, pixel_size, pixel_size)
    
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=num_bands,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(data)
        
    print(f"--> Generated synthetic GeoTIFF: {filename} ({width}x{height} px, {num_bands} bands)")

if __name__ == "__main__":
    # Generate Satellite (10m resolution, 200x200)
    create_dummy_geotiff("data/raw_satellite/sample_sat.tif", width=200, height=200, num_bands=3, pixel_size=10.0)
    
    # Generate Drone (1m resolution, 2000x2000)
    create_dummy_geotiff("data/raw_drone/sample_drone.tif", width=2000, height=2000, num_bands=3, pixel_size=1.0)