from .labelled_dataset import (get_latest_images, find_fire_event_folders, get_corresponding_files)
from .inference_dataset import (find_year_folders, get_matching_images)
from rasterio.windows import Window
import math
import rasterio
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
            year_folders = find_year_folders(input_dir)
            if not year_folders:
                raise ValueError("No year folders found in the inference data source.")
            return year_folders

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
            return get_matching_images(image_path, file_extension=file_extension)

    def _tile_image(self, image_path, output_dir, tile_size=256, file_extension='tiff'):
        """
        Tiles a single image into smaller tiles of specified size and saves them to the output directory.
        Keeps the original channels and metadata intact.

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
                            file_extension='png'  # Save the coverages as PNG (original coverage images are in PNG normally)
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
            
            # Find all year folders in the inference dataset
            year_folders = self._get_folders(self.input_dir)

            if not year_folders:
                print("No year folders found. Please check the input directory.")
                return

            for year_folder in year_folders:
                year_name = os.path.basename(year_folder)
                print(f"\nProcessing year: {year_name}")

                # Create a corresponding output directory for the year
                year_output_dir = os.path.join(tiled_output_dir, year_name)
                os.makedirs(year_output_dir, exist_ok=True)

                # Get all matching Sentinel-1 and Sentinel-2 images for the year
                # This returns a list of dicts, each representing a fire event
                fire_events = get_matching_images(year_folder, file_extension=file_extension)

                if not fire_events:
                    print(f"No images found in {year_folder}")
                    continue
                
                # Loop through each fire event and tile its images
                for event in fire_events:
                    print(f"  Tiling event: {event['id']}")
                    
                    # Tile Sentinel-1 image if it exists
                    if event['sentinel1']:
                        self._tile_image(
                            image_path=event['sentinel1'],
                            output_dir=year_output_dir,
                            tile_size=tile_size,
                            file_extension=file_extension
                        )
                    
                    # Tile Sentinel-2 image if it exists
                    if event['sentinel2']:
                        self._tile_image(
                            image_path=event['sentinel2'],
                            output_dir=year_output_dir,
                            tile_size=tile_size,
                            file_extension=file_extension
                        )

            print(f"\nTiling completed for the inference dataset. Tiled images saved to: {tiled_output_dir}")
        
        else:
            raise ValueError("Invalid data source. Use 'labelled' or 'inference'.")
    
    def visualize_reconstruction(self, tiled_base_dir, fire_event_name=None, year=None, event_id=None, tile_size=256, file_extension='tiff'):
        """
        Reconstructs images from saved tiles and visualizes them.

        Args:
            tiled_base_dir (str): Base directory where the tiled images are saved.
            fire_event_name (str, optional): Name of the fire event (for 'labelled' source). Defaults to None.
            year (str or int, optional): The year of the fire event (for 'inference' source). Defaults to None.
            event_id (str, optional): The unique ID of the fire event (for 'inference' source). Defaults to None.
            tile_size (int, optional): Size of the tiles. Defaults to 256.
            file_extension (str, optional): File extension of the images. Defaults to 'tiff'.
        """

        if self.data_source == 'labelled':

            print(f"\n--- Visualizing reconstruction for: {fire_event_name} ---")

            # 1. Find original files
            original_fire_folder = os.path.join(self.input_dir, fire_event_name)
            if not os.path.exists(original_fire_folder):
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

            # Take Sentinel-2 for visualization if available
            original_image_path = next((p for p in latest_images if 'sentinel2' in p), latest_images[0])
            print(f"Using original image for comparison: {os.path.basename(original_image_path)}")

            # Get corresponding coverage and mask files
            files = self._get_files(original_image_path, file_extension=file_extension)
            original_coverage_path = files.get('coverage')
            original_mask_path = files.get('mask')

            # --- Helper function for reconstruction ---
            def _reconstruct_component(original_path, tiled_subfolder, component_name, tile_ext):
                if not original_path or not os.path.exists(original_path):
                    print(f"Info: Original {component_name} file not found, skipping reconstruction.")
                    return None, None, None

                with rasterio.open(original_path) as src:
                    original_data = src.read()
                    bands, img_height, img_width = original_data.shape
                    tile_height, tile_width = tile_size, tile_size

                tiled_dir = os.path.join(tiled_base_dir, fire_event_name, tiled_subfolder)
                original_basename = os.path.splitext(os.path.basename(original_path))[0]
                
                reconstructed_image = np.zeros_like(original_data, dtype=np.float32)
                counts = np.zeros_like(original_data, dtype=np.float32)
                tile_bboxes = []

                n_tiles_h = math.ceil(img_height / tile_height)
                n_tiles_w = math.ceil(img_width / tile_width)
                y_starts = np.linspace(0, img_height - tile_height, n_tiles_h, dtype=int)
                x_starts = np.linspace(0, img_width - tile_width, n_tiles_w, dtype=int)

                for i, y in enumerate(y_starts):
                    for j, x in enumerate(x_starts):
                        tile_path = os.path.join(tiled_dir, f"{original_basename}_tile_{i}_{j}.{tile_ext}")
                        if os.path.exists(tile_path):
                            with rasterio.open(tile_path) as tile_src:
                                tile = tile_src.read()
                            
                            tile_slice = (slice(None), slice(y, y + tile_height), slice(x, x + tile_width))
                            reconstructed_image[tile_slice] += tile
                            counts[tile_slice] += 1
                            # Bboxes are the same for all components, so only calculate once
                            if component_name == 'image': 
                                tile_bboxes.append((x, y, tile_width, tile_height))
                
                if np.sum(counts) == 0:
                    print(f"ERROR: No tiles found for {component_name} in {tiled_dir}")
                    return None, None, None

                reconstructed_image = np.divide(reconstructed_image, counts, out=np.zeros_like(reconstructed_image), where=counts!=0)
                reconstructed_image = reconstructed_image.astype(original_data.dtype)

                return original_data, reconstructed_image, tile_bboxes

            # 2. Reconstruct all components
            original_img_data, reconstructed_img, tile_bboxes = _reconstruct_component(original_image_path, 'images', 'image', file_extension)
            # Note: Tiled coverages are saved as PNG in the .run() method
            original_cov_data, reconstructed_cov, _ = _reconstruct_component(original_coverage_path, 'coverages', 'coverage', 'png')
            original_mask_data, reconstructed_mask, _ = _reconstruct_component(original_mask_path, 'masks', 'mask', file_extension)

            if original_img_data is None:
                print("Could not reconstruct the main image. Aborting visualization.")
                return

            # 3. Prepare data for plotting
            plot_data = []

            # Prepare Image
            if 'sentinel2' in os.path.basename(original_image_path).lower() and file_extension == 'tiff' and original_img_data.shape[0] >= 4:
                vis_image = np.transpose(original_img_data[[3, 2, 1], :, :], (1, 2, 0))
                vis_reconstructed = np.transpose(reconstructed_img[[3, 2, 1], :, :], (1, 2, 0))
            else:
                vis_image = np.transpose(original_img_data[:3], (1, 2, 0))
                vis_reconstructed = np.transpose(reconstructed_img[:3], (1, 2, 0))
            
            if vis_image.dtype != np.uint8:
                p2, p98 = np.percentile(vis_image, (2, 98))
                if p98 > p2: vis_image = np.clip((vis_image - p2) / (p98 - p2), 0, 1)
                elif vis_image.max() > 0: vis_image = vis_image / vis_image.max()
            
            if vis_reconstructed.dtype != np.uint8:
                p2, p98 = np.percentile(vis_reconstructed, (2, 98))
                if p98 > p2: vis_reconstructed = np.clip((vis_reconstructed - p2) / (p98 - p2), 0, 1)
                elif vis_reconstructed.max() > 0: vis_reconstructed = vis_reconstructed / vis_reconstructed.max()
            
            plot_data.append({'title': 'Image', 'orig': vis_image, 'recon': vis_reconstructed, 'cmap': None})

            # Prepare Coverage
            if original_cov_data is not None:
                plot_data.append({
                    'title': 'Coverage', 
                    'orig': np.squeeze(original_cov_data), 
                    'recon': np.squeeze(reconstructed_cov), 
                    'cmap': 'binary' # Use 'binary' colormap: 0=black, 1/255=white
                })

            # Prepare Mask
            if original_mask_data is not None:
                plot_data.append({
                    'title': 'Fire Mask', 
                    'orig': np.squeeze(original_mask_data), 
                    'recon': np.squeeze(reconstructed_mask), 
                    'cmap': 'viridis'
                })

            # 4. Plotting
            num_rows = len(plot_data)
            fig, axes = plt.subplots(num_rows, 2, figsize=(20, 8 * num_rows), squeeze=False)

            # Generate colors for the grid once
            if tile_bboxes:
                colors = plt.cm.cool(np.linspace(0, 1, len(tile_bboxes)))

            for i, data in enumerate(plot_data):
                # Plot original (left column)
                ax_orig = axes[i, 0]
                ax_orig.imshow(data['orig'], cmap=data['cmap'])
                ax_orig.set_title(f"Original {data['title']}")
                
                # Plot reconstructed (right column)
                ax_recon = axes[i, 1]
                im_recon = ax_recon.imshow(data['recon'], cmap=data['cmap'])
                ax_recon.set_title(f"Reconstructed {data['title']}")

                # Add grid overlay ONLY to the original plot (first column)
                if tile_bboxes:
                    ax_orig.set_title(ax_orig.get_title() + " with Tile Grid Overlay")
                    for k, (x, y, w, h) in enumerate(tile_bboxes):
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[k], facecolor='none', alpha=0.8)
                        ax_orig.add_patch(rect)

                # Add legends to reconstructed plots
                if data['title'] == 'Coverage':
                    legend_patches = [patches.Patch(color='white', label='Valid'),
                                    patches.Patch(color='black', label='Invalid')]
                    ax_recon.legend(handles=legend_patches, loc='best')
                
                elif data['title'] == 'Fire Mask':
                    # Use make_axes_locatable to prevent misalignment
                    divider = make_axes_locatable(ax_recon)
                    cax = divider.append_axes("right", size="1%", pad=0.1)  # Change size of the color bar if images are not aligned
                    cbar = fig.colorbar(im_recon, cax=cax)
                    # cbar.set_label('Fire Severity Level')

                # Remove ticks from all axes
                for ax in axes[i]:
                    ax.set_xticks([]); ax.set_yticks([])
            
            plt.tight_layout()
            plt.show()

        elif self.data_source == 'inference':
            print(f"\n--- Visualizing reconstruction for inference data ---")
        
            if not year or not event_id:
                print("ERROR: For inference data, 'year' and 'event_id' must be provided.")
                return

            # --- 1. Find original and tiled files ---
            original_year_folder = os.path.join(self.input_dir, str(year))
            tiled_year_folder = os.path.join(tiled_base_dir, str(year))

            if not os.path.exists(original_year_folder):
                print(f"ERROR: Original year folder not found at {original_year_folder}")
                return
            if not os.path.exists(tiled_year_folder):
                print(f"ERROR: Tiled year folder not found at {tiled_year_folder}")
                return

            # Find the specific event from the original data to get file paths
            all_events = get_matching_images(original_year_folder, file_extension=file_extension)
            target_event = next((e for e in all_events if e['id'] == event_id), None)

            if not target_event:
                print(f"ERROR: Event with ID '{event_id}' not found in year {year}.")
                return
                
            original_s1_path = target_event.get('sentinel1')
            original_s2_path = target_event.get('sentinel2')
            
            print(f"Found event: {event_id} in year {year}")
            if original_s1_path: print(f"  Original S1: {os.path.basename(original_s1_path)}")
            if original_s2_path: print(f"  Original S2: {os.path.basename(original_s2_path)}")

            # --- Helper function for reconstruction ---
            def _reconstruct_inference_component(original_path, tiled_dir, component_name):
                if not original_path or not os.path.exists(original_path):
                    print(f"Info: Original {component_name} file not found, skipping reconstruction.")
                    return None, None, None

                with rasterio.open(original_path) as src:
                    original_data = src.read()
                    bands, img_height, img_width = original_data.shape
                    tile_height, tile_width = tile_size, tile_size

                original_basename = os.path.splitext(os.path.basename(original_path))[0]
                
                reconstructed_image = np.zeros_like(original_data, dtype=np.float32)
                counts = np.zeros_like(original_data, dtype=np.float32)
                tile_bboxes = []

                n_tiles_h = math.ceil(img_height / tile_height)
                n_tiles_w = math.ceil(img_width / tile_width)
                y_starts = np.linspace(0, img_height - tile_height, n_tiles_h, dtype=int)
                x_starts = np.linspace(0, img_width - tile_width, n_tiles_w, dtype=int)

                for i, y in enumerate(y_starts):
                    for j, x in enumerate(x_starts):
                        tile_path = os.path.join(tiled_dir, f"{original_basename}_tile_{i}_{j}.{file_extension}")
                        if os.path.exists(tile_path):
                            with rasterio.open(tile_path) as tile_src:
                                tile = tile_src.read()
                            
                            tile_slice = (slice(None), slice(y, y + tile_height), slice(x, x + tile_width))
                            reconstructed_image[tile_slice] += tile
                            counts[tile_slice] += 1
                            tile_bboxes.append((x, y, tile_width, tile_height))
                
                if np.sum(counts) == 0:
                    print(f"ERROR: No tiles found for {component_name} in {tiled_dir}")
                    return None, None, None

                reconstructed_image = np.divide(reconstructed_image, counts, out=np.zeros_like(reconstructed_image), where=counts!=0)
                reconstructed_image = reconstructed_image.astype(original_data.dtype)

                return original_data, reconstructed_image, tile_bboxes

            # --- 2. Reconstruct S1 and S2 images ---
            original_s1_data, reconstructed_s1, s1_bboxes = _reconstruct_inference_component(original_s1_path, tiled_year_folder, 'Sentinel-1')
            original_s2_data, reconstructed_s2, s2_bboxes = _reconstruct_inference_component(original_s2_path, tiled_year_folder, 'Sentinel-2')

            if original_s1_data is None and original_s2_data is None:
                print("Could not reconstruct any image. Aborting visualization.")
                return

            # --- 3. Plotting ---
            plot_data = []
            if original_s1_data is not None:
                # For S1, visualize the first band (VV polarization) in grayscale
                plot_data.append({
                    'title': 'Sentinel-1 (VV Band)',
                    'orig': np.squeeze(original_s1_data[0, :, :]),
                    'recon': np.squeeze(reconstructed_s1[0, :, :]),
                    'cmap': 'gray',
                    'bboxes': s1_bboxes
                })
            
            if original_s2_data is not None:
                # For S2, visualize RGB (Bands B04, B03, B02 are at indices 3, 2, 1)
                vis_image = np.transpose(original_s2_data[[3, 2, 1], :, :], (1, 2, 0))
                vis_reconstructed = np.transpose(reconstructed_s2[[3, 2, 1], :, :], (1, 2, 0))
                
                # Normalize for better visualization
                p2, p98 = np.percentile(vis_image, (2, 98))
                if p98 > p2: vis_image = np.clip((vis_image - p2) / (p98 - p2), 0, 1)
                elif vis_image.max() > 0: vis_image = vis_image / vis_image.max()
                
                p2, p98 = np.percentile(vis_reconstructed, (2, 98))
                if p98 > p2: vis_reconstructed = np.clip((vis_reconstructed - p2) / (p98 - p2), 0, 1)
                elif vis_reconstructed.max() > 0: vis_reconstructed = vis_reconstructed / vis_reconstructed.max()

                plot_data.append({
                    'title': 'Sentinel-2 (RGB)',
                    'orig': vis_image,
                    'recon': vis_reconstructed,
                    'cmap': None,
                    'bboxes': s2_bboxes
                })

            num_rows = len(plot_data)
            if num_rows == 0:
                print("No data to plot.")
                return
                
            fig, axes = plt.subplots(num_rows, 2, figsize=(20, 8 * num_rows), squeeze=False)
            fig.suptitle(f"Reconstruction for Event: {event_id} (Year: {year})", fontsize=20)

            for i, data in enumerate(plot_data):
                ax_orig = axes[i, 0]
                ax_orig.imshow(data['orig'], cmap=data['cmap'])
                ax_orig.set_title(f"Original {data['title']} with Tile Grid")
                
                ax_recon = axes[i, 1]
                ax_recon.imshow(data['recon'], cmap=data['cmap'])
                ax_recon.set_title(f"Reconstructed {data['title']}")

                if data['bboxes']:
                    colors = plt.cm.cool(np.linspace(0, 1, len(data['bboxes'])))
                    for k, (x, y, w, h) in enumerate(data['bboxes']):
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[k], facecolor='none', alpha=0.8)
                        ax_orig.add_patch(rect)

                for ax in axes[i]:
                    ax.set_xticks([]); ax.set_yticks([])
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        else:
            raise ValueError("Invalid data source. Use 'labelled' or 'inference'.")