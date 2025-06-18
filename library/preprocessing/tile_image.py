import os
import rasterio
from rasterio.windows import Window

def tile_image(file_path, output_dir, tile_size=512, file_type='image'):
    """Create tiles of size tile_size x tile_size from the image"""
    try:
        # Create output directory if it doesn't exist
        fire_name = os.path.basename(os.path.dirname(file_path))
        image_name = os.path.basename(file_path).split('.')[0]
        
        # Make sure the output directory exists
        specific_output_dir = os.path.join(output_dir, fire_name, file_type)
        os.makedirs(specific_output_dir, exist_ok=True)
        
        with rasterio.open(file_path) as src:
            # Get image dimensions
            height = src.height
            width = src.width
            
            print(f"Tiling {file_path} ({width}x{height}) into {tile_size}x{tile_size} tiles")
            
            # Calculate number of tiles in each dimension
            num_tiles_h = height // tile_size + (1 if height % tile_size > 0 else 0)
            num_tiles_w = width // tile_size + (1 if width % tile_size > 0 else 0)
            
            print(f"Will create up to {num_tiles_h}x{num_tiles_w}={num_tiles_h*num_tiles_w} tiles")
            tiles_created = 0
            
            # Extract and save tiles
            for i in range(num_tiles_h):
                for j in range(num_tiles_w):
                    # Calculate tile window
                    y_offset = i * tile_size
                    x_offset = j * tile_size
                    
                    # Handle edge cases (partial tiles at edges)
                    window_height = min(tile_size, height - y_offset)
                    window_width = min(tile_size, width - x_offset)
                    
                    window = Window(x_offset, y_offset, window_width, window_height)
                    
                    # Read the tile
                    tile_data = src.read(window=window)
                    
                    # Only save full-size tiles
                    if window_height == tile_size and window_width == tile_size:
                        tile_filename = f"{image_name}_tile_{i}_{j}.tiff"
                        tile_path = os.path.join(specific_output_dir, tile_filename)
                        
                        # Create a new raster file with the tile data
                        with rasterio.open(
                            tile_path,
                            'w',
                            driver='GTiff',
                            height=tile_size,
                            width=tile_size,
                            count=src.count,
                            dtype=tile_data.dtype,
                            crs=src.crs,
                            transform=src.window_transform(window),
                        ) as dst:
                            dst.write(tile_data)
                            tiles_created += 1
            
            print(f"Created {tiles_created} tiles for {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error while tiling {file_path}: {str(e)}")