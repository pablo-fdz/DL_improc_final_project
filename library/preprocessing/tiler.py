from library.preprocessing.labelled_dataset import (get_latest_images, 
                                                    find_fire_event_folders,
                                                    get_corresponding_files)
from rasterio.windows import Window
import math
import rasterio
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Tiler():
    """
    Class to handle tiling of images from a data source.
    """
    def __init__(self, data_source, input_dir):
        """
        Initializes the Tiler with a data source and input directory.
        Args:
            data_source (str): Type of data source ('labelled' for the training dataset,
                'inference' for the inference dataset).
            input_dir (str): Root directory containing the images to be tiled.
        """
        self.data_source = data_source
        self.input_dir = input_dir
    
    def _get_folders(self, input_dir):
        """
        Retrieves the folders containing fire event images based on the data source.

        Returns:
            list: List of folder paths containing fire event images.
        """
        if self.data_source == 'labelled':
            fire_folders = find_fire_event_folders(input_dir)
            if not fire_folders:
                raise ValueError("No fire event folders found.")
            return fire_folders
        
        elif self.data_source == 'inference':
            raise ValueError("Inference data source does not have fire event folders.")

    def _get_latest_images(self, folder, file_extension='tiff'):
        """
        Retrieves the latest images from a given folder.

        Args:
            folder (str): Path to the folder containing images.
            file_extension (str): File extension of the images to be processed (default is 'tiff').

        Returns:
            list: List of file paths to the latest images.
        """

        if self.data_source == 'labelled':
            # For labelled data, we assume the folder contains fire event images
            return get_latest_images(folder, file_extension=file_extension)
        
        elif self.data_source == 'inference':
            raise ValueError("Inference data source only has post-fire images.")

    def _get_files(self, image_path, file_extension='tiff'):
        """
        Retrieves the corresponding files (coverage and mask) for a given image path.

        Args:
            image_path (str): Path to the image file.
            file_extension (str): File extension of the images to be extracted (default is 'tiff').

        Returns:
            dict: Dictionary containing paths to the image, coverage, and mask files.
        """

        if self.data_source == 'labelled':
            # For labelled data, we assume the image path is part of a fire event folder
            return get_corresponding_files(image_path, file_extension=file_extension)
        
        elif self.data_source == 'inference':
            raise ValueError("Inference data source does not have corresponding files.")

    def _tile_image(self, image_path, output_dir, tile_size=256, file_extension='tiff'):
        """
        Tiles a single image into smaller tiles of specified size and saves them to the output directory.

        Args:
            image_path (str): Path to the image file to be tiled.
            output_dir (str): Directory where the tiled images will be saved.
            tile_size (int): Size of the tiles to be created (default is 256).
            file_extension (str): File extension for the output files (default is 'tiff').
        """

        try:
            with rasterio.open(image_path) as src:
                image_data = src.read()
                source_transform = src.transform
                source_crs = src.crs
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                bands, img_height, img_width = image_data.shape
                tile_height, tile_width = tile_size, tile_size

                if img_height < tile_height or img_width < tile_width:
                    print(f"    Skipping {base_name}, image smaller than tile size.")
                    return

                n_tiles_h = math.ceil(img_height / tile_height)
                n_tiles_w = math.ceil(img_width / tile_width)
                
                y_starts = np.linspace(0, img_height - tile_height, n_tiles_h, dtype=int)
                x_starts = np.linspace(0, img_width - tile_width, n_tiles_w, dtype=int)

                tile_count = 0
                for i, y in enumerate(y_starts):
                    for j, x in enumerate(x_starts):
                        window = Window(x, y, tile_width, tile_height)
                        tile_data = src.read(window=window)
                        
                        # Define output filename with row and column index
                        tile_filename = f"{base_name}_tile_{i}_{j}.{file_extension}"
                        tile_path = os.path.join(output_dir, tile_filename)
                        
                        profile = src.profile
                        profile.update({
                            'height': tile_height,
                            'width': tile_width,
                            'transform': src.window_transform(window)
                        })

                        # Use rasterio.Env to disable creation of .aux.xml files
                        with rasterio.Env(GDAL_PAM_ENABLED='NO'):
                            with rasterio.open(tile_path, 'w', **profile) as dst:
                                dst.write(tile_data)
                        tile_count += 1
                
                print(f"    Generated {tile_count} tiles for {os.path.basename(image_path)}")

        except Exception as e:
            print(f"    ERROR tiling {image_path}: {e}")

    def run(self, tiled_output_dir, file_extension='tiff', tile_size=256):
        """
        Finds all fire events and tiles their latest images, coverages, and masks.

        Args:
            tiled_output_dir (str): Root directory where the tiled images will be saved.
            file_extension (str): File extension of the images to be processed (default is 'tiff').
            tile_size (int): Size of the tiles to be created (default is 256).

        Returns:
            None
        """

        # Create base name for output directory
        tiled_output_dir = os.path.join(tiled_output_dir, f'tiled_{self.data_source}_dataset')

        if self.data_source == 'labelled':

            # Find all fire event folders
            fire_folders = self._get_folders(self.input_dir)
            
            if not fire_folders:
                print("No fire event folders found. Please check the base path.")
                return
            
            for folder in fire_folders:

                print(f"\nProcessing folder: {folder}")
                
                # Get the name of the fire event from the folder path (last part of the path)
                fire_event_name = os.path.basename(folder)
                fire_dir = os.path.join(tiled_output_dir, fire_event_name)
                os.makedirs(fire_dir, exist_ok=True)  # Ensure output directory exists

                # Get the latest images 
                latest_images = self._get_latest_images(folder, file_extension=file_extension)  # This will return a list of the latest Sentinel-1 and Sentinel-2 images
                
                if not latest_images:
                    print(f"No suitable images found in {folder}")
                    continue
                
                # For each fire, loop over the Sentinel-1 and Sentinel-2 images
                for image_path in latest_images:  
                    # Get corresponding files associated to the images
                    files = self._get_files(image_path, file_extension=file_extension)
                    
                    # Tile the image
                    print(f"Tiling image: {files['image']}")
                    # Create output directory for the image
                    img_output_dir = os.path.join(fire_dir, 'images')
                    os.makedirs(img_output_dir, exist_ok=True)
                    # Tile the image and save to the output directory
                    self._tile_image(
                        image_path=files['image'],
                        output_dir=img_output_dir,
                        tile_size=tile_size,
                        file_extension=file_extension
                    )
                    
                    # Tile the coverage if available
                    if files['coverage']:
                        print(f"Tiling coverage: {files['coverage']}")
                        # Create output directory for the coverage
                        coverage_output_dir = os.path.join(fire_dir, 'coverages')
                        os.makedirs(coverage_output_dir, exist_ok=True)
                        # Tile the coverage and save to the output directory
                        self._tile_image(
                            image_path=files['coverage'],
                            output_dir=coverage_output_dir,
                            tile_size=tile_size,
                            file_extension=file_extension
                        )
                    
                    # Tile the mask if available
                    if files['mask']:
                        print(f"Tiling mask: {files['mask']}")
                        # Create output directory for the mask
                        mask_output_dir = os.path.join(fire_dir, 'masks')
                        os.makedirs(mask_output_dir, exist_ok=True)
                        # Tile the mask and save to the output directory
                        self._tile_image(
                            image_path=files['mask'],
                            output_dir=mask_output_dir,
                            tile_size=tile_size,
                            file_extension=file_extension
                        )
                
                print(f"Finished processing folder: {folder}")

            print(f"\nTiling completed for all fire events. Tiled images saved to: {tiled_output_dir}")

        elif self.data_source == 'inference':
            
            print("Not implemented yet for inference data source.")
        
        else:
            raise ValueError("Invalid data source. Use 'labelled' or 'inference'.")
    
    def visualize_reconstruction(self, fire_event_name, tiled_base_dir, tile_size=256, file_extension='tiff'):
        """
        Reconstructs an image from its saved tiles and visualizes the original vs. reconstructed.

        Args:
            fire_event_name (str): Name of the fire event to visualize.
            tiled_base_dir (str): Base directory where the tiled images are saved.
            tile_size (int): Size of the tiles used for reconstruction (default is 256).
            file_extension (str): File extension of the images to be visualized (default is 'tiff').
        """
        print(f"\n--- Visualizing reconstruction for: {fire_event_name} ---")
        
        # 1. Find the original image to get its dimensions and for comparison
        original_fire_folder = os.path.join(self.input_dir, fire_event_name)
        if not os.path.exists(original_fire_folder):
             # Try finding it in one of the part folders
            found = False
            for part in range(1, 6):
                path = os.path.join(self.input_dir, f"Satellite_burned_area_dataset_part{part}", fire_event_name)
                if os.path.exists(path):
                    original_fire_folder = path
                    found = True
                    break
            if not found:
                print(f"ERROR: Could not find original folder for '{fire_event_name}'")
                return

        latest_images = self._get_latest_images(original_fire_folder, file_extension=file_extension)
        if not latest_images:
            print("ERROR: No original images found to compare against.")
            return
        
        # Prefer Sentinel-2 for visualization
        original_image_path = next((p for p in latest_images if 'sentinel2' in p), latest_images[0])
        print(f"Using original image for comparison: {os.path.basename(original_image_path)}")

        # 2. Load original image and prepare for visualization
        with rasterio.open(original_image_path) as src:
            original_data = src.read()
            vis_image = np.transpose(original_data[:3], (1, 2, 0))
            if vis_image.max() > 1.0 and vis_image.dtype != np.uint8:
                vis_image = vis_image / vis_image.max()

        bands, img_height, img_width = original_data.shape
        tile_height, tile_width = tile_size, tile_size

        # 3. Find all corresponding tiles
        tiled_image_dir = os.path.join(tiled_base_dir, fire_event_name, 'images')
        original_basename = os.path.splitext(os.path.basename(original_image_path))[0]
        tile_paths = glob.glob(os.path.join(tiled_image_dir, f"{original_basename}_tile_*.{file_extension}"))

        if not tile_paths:
            print(f"ERROR: No tiles found for {original_basename} in {tiled_image_dir}")
            return

        # 4. Reconstruct from saved tiles
        reconstructed_image = np.zeros_like(original_data, dtype=np.float32)
        counts = np.zeros_like(original_data, dtype=np.float32)
        tile_bboxes = []

        n_tiles_h = math.ceil(img_height / tile_height)
        n_tiles_w = math.ceil(img_width / tile_width)
        y_starts = np.linspace(0, img_height - tile_height, n_tiles_h, dtype=int)
        x_starts = np.linspace(0, img_width - tile_width, n_tiles_w, dtype=int)

        for i, y in enumerate(y_starts):
            for j, x in enumerate(x_starts):
                tile_path = os.path.join(tiled_image_dir, f"{original_basename}_tile_{i}_{j}.{file_extension}")
                if os.path.exists(tile_path):
                    with rasterio.open(tile_path) as tile_src:
                        tile = tile_src.read()
                    
                    tile_slice = (slice(None), slice(y, y + tile_height), slice(x, x + tile_width))
                    reconstructed_image[tile_slice] += tile
                    counts[tile_slice] += 1
                    tile_bboxes.append((x, y, tile_width, tile_height))

        # Average the pixels in overlapping regions
        reconstructed_image = np.divide(reconstructed_image, counts, out=np.zeros_like(reconstructed_image), where=counts!=0)
        reconstructed_image = reconstructed_image.astype(original_data.dtype)

        # Prepare reconstructed image for visualization
        vis_reconstructed = np.transpose(reconstructed_image[:3], (1, 2, 0))
        if vis_reconstructed.max() > 1.0 and vis_reconstructed.dtype != np.uint8:
            vis_reconstructed = vis_reconstructed / vis_reconstructed.max()

        # 5. Plotting
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(vis_image)
        axes[0].set_title('Original Image with Tile Grid Overlay')
        colors = plt.cm.cool(np.linspace(0, 1, len(tile_bboxes)))
        for k, (x, y, w, h) in enumerate(tile_bboxes):
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[k], facecolor='none', alpha=1)
            axes[0].add_patch(rect)
            axes[0].text(x + 5, y + 20, f'{k+1}', fontsize=8, color='white', bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[k], alpha=1))

        axes[1].imshow(vis_reconstructed)
        axes[1].set_title('Reconstructed Image from Tiles')
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()