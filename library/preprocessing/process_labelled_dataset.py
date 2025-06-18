import os
from .tile_image import tile_image
from .labelled_dataset import (get_latest_tiff_images, find_fire_event_folders,
                              get_corresponding_files)

def process_labelled_dataset(base_path, output_dir, tile_size=512):
    """
    Main function to process the dataset
    
    Args:
        base_path (str): Path to the dataset or a specific part of it
        output_dir (str): Directory where the output tiles will be saved
        tile_size (int): Size of the tiles to create (default is 512x512 pixels)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting to process dataset at {base_path}")
    print(f"Output will be saved to {output_dir}")
    
    # Find all fire event folders
    fire_folders = find_fire_event_folders(base_path)
    
    if not fire_folders:
        print("No fire event folders found. Please check the base path.")
        return
    
    for folder in fire_folders:
        print(f"\nProcessing folder: {folder}")
        
        # Get the 2 latest TIFF images
        latest_images = get_latest_tiff_images(folder)
        
        if not latest_images:
            print(f"No suitable TIFF images found in {folder}")
            continue
        
        for image_path in latest_images:
            # Get corresponding files
            files = get_corresponding_files(image_path)
            
            # Tile the image
            print(f"Tiling image: {files['image']}")
            tile_image(files['image'], output_dir, tile_size=tile_size, file_type='images')
            
            # Tile the coverage if available
            if files['coverage']:
                print(f"Tiling coverage: {files['coverage']}")
                tile_image(files['coverage'], output_dir, tile_size=tile_size, file_type='coverages')
            
            # Tile the mask if available
            if files['mask']:
                print(f"Tiling mask: {files['mask']}")
                tile_image(files['mask'], output_dir, tile_size=tile_size, file_type='masks')