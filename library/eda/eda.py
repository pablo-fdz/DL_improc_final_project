from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import tqdm
import numpy as np
import torch
from collections import defaultdict
from ..utilities.read_tiff_to_torch import read_tiff_to_torch
from ..utilities.read_png_to_torch import read_png_to_torch
from typing import List, Dict
import tifffile
import os


class DataExplorer:
    def __init__(self, list_of_paths: Dict[str, List[str]]):
        """
        Class holding the data exploration functions and metrics/information for a specific type of images.
        To be initialized with a dictionary of lists of paths to the images of each category (i.e. masks, pre-images, post-images).
        """

        self.paths = list_of_paths
        self.pixel_values_dict = {0.0: '0',
                                  32.0: '1',
                                  64.0: '2',
                                  128.0: '3',
                                  192.0: '4',
                                  255.0: '5'
                                  }

        # ATTRIBUTES TO BE DETERMINED LATER
        self.pixel_counts = defaultdict(int)
        self.total_pixel_count = 0
        self.images_with_only_zero = 0

        self.coverage_pixel_counts = defaultdict(int)
        self.total_coverage_pixels = 0
        self.images_with_only_white = 0
        self.images_with_black_pixels = 0
        self.white_pixels_in_defective = 0
        self.black_pixels_in_defective = 0

        self.defective_coverages_path_list = []
        self.files_with_only_0_labels_path_list = []

    def count_coverage_pixels(self, dtype: torch.dtype = torch.int16, normalize: bool = False):
        """Count coverage pixels to identify defective areas"""
        file_paths = self.paths.get('coverages', [])

        for path in file_paths:
            try:
                img = read_png_to_torch(path, dtype=dtype, normalize=normalize)
                total_pixels = img.numel()
                self.total_coverage_pixels += total_pixels

                values, counts = torch.unique(flat, return_counts=True)

                # Check pixel composition
                has_black = 0 in values
                has_white = 255 in values
                
                if not has_black and has_white:
                    # Only white pixels
                    self.images_with_only_white += 1
                elif has_black:
                    # Has some black pixels (defective)
                    self.images_with_black_pixels += 1
                    self.defective_coverages_path_list.append(path)
                    
                    # Count white and black pixels in this defective image
                    for val, count in zip(values, counts):
                        if val.item() == 0:  # Black pixels
                            self.black_pixels_in_defective += count.item()
                        elif val.item() == 255:  # White pixels
                            self.white_pixels_in_defective += count.item()

                # Overall pixel counts
                for val, count in zip(values, counts):
                    self.coverage_pixel_counts[val.item()] += count.item()

            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        # Print results
        print("Coverage Analysis:")
        print(f"Images with only white pixels: {self.images_with_only_white}")
        print(f"Images with black pixels (defective): {self.images_with_black_pixels}")
        
        if self.images_with_black_pixels > 0:
            total_defective_pixels = self.white_pixels_in_defective + self.black_pixels_in_defective
            white_pct = (self.white_pixels_in_defective / total_defective_pixels) * 100
            black_pct = (self.black_pixels_in_defective / total_defective_pixels) * 100
            print(f"In defective images - White: {white_pct:.2f}%, Black: {black_pct:.2f}%")

    def visualize_coverage_pixels(self):
        """Visualize coverage pixel distribution with horizontal bar plots"""
        if not hasattr(self, 'coverage_pixel_counts'):
            print("No coverage pixel data. Run count_coverage_pixels() first.")
            return

        # Get viridis colors
        viridis_colors = self._get_viridis_colors_for_classes(self.pixel_values_dict)
        selected_colors = [viridis_colors[1], viridis_colors[4]]  # Use 2nd and 5th colors

        fig, axes = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [1, 1]})

        sns.set(style="whitegrid", palette="pastel", font_scale=1.2)

        # Top plot: Images with only white vs images with some black pixels
        ax1 = axes[0]
        image_counts = [self.images_with_only_white, self.images_with_black_pixels]
        image_labels = ['Only white pixels\n(valid coverage)', 'Some black pixels\n(defective areas)']
        
        sns.barplot(
            x=image_counts,
            y=image_labels,
            palette=selected_colors,
            orient='h',
            ax=ax1
        )
        ax1.set_xlabel('Number of Images')
        ax1.set_ylabel('')
        ax1.set_title('Coverage Image Quality Distribution')
        
        # Add count annotations
        for p in ax1.patches:
            ax1.annotate(f'{int(p.get_width())}', 
                        (p.get_width(), p.get_y() + p.get_height() / 2.), 
                        ha='left', va='center', fontsize=11, color='black', 
                        xytext=(5, 0), textcoords='offset points')
        
        sns.despine(ax=ax1)

        # Bottom plot: Pixel distribution in defective images only
        ax2 = axes[1]
        
        if self.images_with_black_pixels > 0:
            pixel_counts = [self.white_pixels_in_defective, self.black_pixels_in_defective]
            pixel_labels = ['White pixels\n(valid areas)', 'Black pixels\n(defective areas)']
            
            # Calculate percentages
            total_defective_pixels = sum(pixel_counts)
            percentages = [(count/total_defective_pixels)*100 for count in pixel_counts]
            
            sns.barplot(
                x=pixel_counts,
                y=pixel_labels,
                palette=selected_colors,
                orient='h',
                ax=ax2
            )
            ax2.set_xlabel('Number of Pixels')
            ax2.set_ylabel('')
            ax2.set_title('Pixel Distribution in Defective Coverage Images')
            
            # Add count and percentage annotations
            for p, pct in zip(ax2.patches, percentages):
                ax2.annotate(f'{int(p.get_width()):,}\n({pct:.1f}%)', 
                            (p.get_width(), p.get_y() + p.get_height() / 2.), 
                            ha='left', va='center', fontsize=10, color='black', 
                            xytext=(5, 0), textcoords='offset points')
        else:
            ax2.text(0.5, 0.5, 'No defective images found', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_title('Pixel Distribution in Defective Coverage Images')
        
        sns.despine(ax=ax2)
        plt.tight_layout()
        plt.show()

    def count_mask_pixels(self, dtype: torch.dtype = torch.float32, normalize: bool = False):
        file_paths = self.paths.get('masks', [])
        
        self.pixel_counts = defaultdict(int)
        self.total_pixel_count = 0
        self.images_with_only_zero = 0

        for path in file_paths:
            img = read_tiff_to_torch(path, dtype=dtype, normalize=normalize)
            total_pixels = img.numel()
            self.total_pixel_count += total_pixels

            flat = img.flatten()
            values, counts = torch.unique(flat, return_counts=True)

            # Check if this image has only 0.0 pixels
            if len(values) == 1 and values[0].item() == 0.0:
                self.images_with_only_zero += 1
                self.files_with_only_0_labels_path_list.append(path)

            for val, count in zip(values, counts):
                self.pixel_counts[val.item()] += count.item()

        # Print final aggregated results
        for val, count in sorted(self.pixel_counts.items()):
            pct = (count / self.total_pixel_count) * 100
            print(f"Value {val:<5}: {count:>11,} pixels ({pct:5.2f} %)")

        total_images = len(file_paths)
        zero_pct = (self.images_with_only_zero / total_images) * 100 if total_images > 0 else 0

        print(f"\nImages with only negative pixels: {self.images_with_only_zero} ({zero_pct:5.2f}%)")

    def _get_viridis_colors_for_classes(self, pixel_values_dict):
        """Get viridis colors that match the order of classes in pixel_values_dict"""
        pixel_values = list(pixel_values_dict.keys())  # [0.0, 32.0, 64.0, 128.0, 192.0, 255.0]
        
        viridis = plt.cm.viridis
        vmin, vmax = min(pixel_values), max(pixel_values)
            
        colors = []
        for val in pixel_values:
            normalized = (val - vmin) / (vmax - vmin) if vmax != vmin else 0
            rgba = viridis(normalized)
            hex_color = mcolors.to_hex(rgba)
            colors.append(hex_color)
            
        return colors
    
    def count_regrouped_mask_pixels(self, dtype: torch.dtype = torch.float32, normalize: bool = False, normalization_method: str = None,
                        map_to_classes: bool = False):
        """
        Count mask pixels with option to map to 5 burn severity classes
        
        Args:
            dtype: Data type for the tensor
            normalize: Whether to normalize the pixel values
            map_to_classes: If True, map pixel values to 5 burn severity classes 
                        representing [0, 64, 128, 191, 255] in the original scale
        """
        file_paths = self.paths.get('masks', [])
        
        self.pixel_counts = defaultdict(int)
        self.total_pixel_count = 0
        self.images_with_only_zero = 0
        self.files_with_only_0_labels_path_list = []
        
        # Update pixel values dictionary if mapping to 5 classes
        if map_to_classes:
            self.pixel_values_dict = {
                0: '0',
                1: '1',
                2: '2',
                3: '3',
                4: '4'
            }

        for path in file_paths:
            img = read_tiff_to_torch(path, dtype=dtype, normalize=normalize, normalization_method=normalization_method)
            
            # Apply mapping to classes if requested
            if map_to_classes:
                img = self._map_to_burn_severity_classes(img)
            
            total_pixels = img.numel()
            self.total_pixel_count += total_pixels

            flat = img.flatten()
            values, counts = torch.unique(flat, return_counts=True)

            # Check if this image has only 0 pixels
            if len(values) == 1 and values[0].item() == 0:
                self.images_with_only_zero += 1
                self.files_with_only_0_labels_path_list.append(path)

            for val, count in zip(values, counts):
                self.pixel_counts[val.item()] += count.item()

        # Print final aggregated results
        for val, count in sorted(self.pixel_counts.items()):
            pct = (count / self.total_pixel_count) * 100
            class_label = self.pixel_values_dict.get(val, f"Unknown ({val})")
            print(f"Class {class_label:<20}: {count:>11,} pixels ({pct:5.2f} %)")

        total_images = len(file_paths)
        zero_pct = (self.images_with_only_zero / total_images) * 100 if total_images > 0 else 0

        print(f"\nImages with only unburned pixels: {self.images_with_only_zero} ({zero_pct:5.2f}%)")

    def _map_to_burn_severity_classes(self, img):
        """
        Map original pixel values (0-255) to 5 burn severity classes (0-4)
        """
        # Create output tensor with same shape but integer type
        mapped = torch.zeros_like(img, dtype=torch.long)
        
        # Define class boundaries in 0-255 space
        mapped[(img <= 32)] = 0                         # Unburned
        mapped[(img > 32) & (img <= 96)] = 1            # Low Severity
        mapped[(img > 96) & (img <= 159)] = 2           # Moderate-Low Severity
        mapped[(img > 159) & (img <= 223)] = 3          # Moderate-High Severity
        mapped[(img > 223)] = 4                         # High Severity
        
        return mapped
    
    def visualize_mask_pixels(self):

        # Main pixel count barplot (sorted by class label order)
        ordered_pixel_values = list(self.pixel_values_dict.keys())
        ordered_class_labels = [self.pixel_values_dict[v] for v in ordered_pixel_values]
        counts = [self.pixel_counts.get(v, 0) for v in ordered_pixel_values]
        
        # Get viridis colors in the same order
        viridis_colors = self._get_viridis_colors_for_classes(self.pixel_values_dict)
        reduced_viridis = viridis_colors[1], viridis_colors[4] # second and fifth colors

        fig, axes = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [3, 1]})

        sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
        ax = axes[0]
        
        # Use viridis colors instead of Blues_d
        sns.barplot(x=ordered_class_labels, y=counts, palette=viridis_colors, ax=ax)

        ax.set_xlabel('Class Label', fontsize=12)
        ax.set_ylabel('Pixel Count', fontsize=12)
        ax.set_title('Pixel Counts per Class', fontsize=14)

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height()):,}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10, color='black', rotation=0, xytext=(0, 5), textcoords='offset points')

        sns.despine(ax=ax)

        # HORIZONTAL BARPLOT FOR IMAGE TYPES
        total_images = len(self.paths.get('masks', []))
        only_negative = self.images_with_only_zero
        negative_and_positive = total_images - only_negative

        ax2 = axes[1]
        sns.barplot(
            x=[only_negative, negative_and_positive],
            y=['Only negative', 'Negative & positive'],
            palette=reduced_viridis,
            orient='h',
            ax=ax2
        )
        ax2.set_xlabel('Number of Images')
        ax2.set_ylabel('')
        ax2.set_title('Image Types by Pixel Content')
        for p in ax2.patches:
            ax2.annotate(f'{int(p.get_width())}', 
                        (p.get_width(), p.get_y() + p.get_height() / 2.), 
                        ha='left', va='center', fontsize=10, color='black', xytext=(5, 0), textcoords='offset points')
        sns.despine(ax=ax2)

        plt.tight_layout()
        plt.show()


    def plot_sentinel2_bands_distribution(self):
        """
        Creates a density plot for all 13 Sentinel-2 bands on a single axis,
        using colors that approximate the actual wavelengths where possible.
        """
        
        sentinel2_paths = self.paths.get('sentinel2', [])
        
        if not sentinel2_paths:
            print("No Sentinel-2 images found")
            return
        
        # Band names and their wavelengths
        band_info = {
            0: "B1 - Coastal aerosol (443 nm)",
            1: "B2 - Blue (490 nm)",
            2: "B3 - Green (560 nm)",
            3: "B4 - Red (665 nm)",
            4: "B5 - Vegetation Red Edge (705 nm)",
            5: "B6 - Vegetation Red Edge (740 nm)",
            6: "B7 - Vegetation Red Edge (783 nm)",
            7: "B8 - NIR (842 nm)",
            8: "B8A - Narrow NIR (865 nm)",
            9: "B9 - Water vapor (945 nm)",
            10: "B10 - SWIR Cirrus (1375 nm)",
            11: "B11 - SWIR (1610 nm)",
            12: "B12 - SWIR (2190 nm)"
        }
        
        # Define colors that approximate the actual wavelengths
        band_colors = {
            0: '#6a92ff',  # Light blue for coastal aerosol
            1: '#0000ff',  # Blue
            2: '#00ff00',  # Green
            3: '#ff0000',  # Red
            4: '#ff3d3d',  # Lighter red for red edge
            5: '#ff7d7d',  # Pale red for red edge
            6: '#ffb1b1',  # Very pale red for red edge
            7: '#8f1fff',  # Purple for NIR
            8: '#7300ff',  # Deeper purple for narrow NIR
            9: '#4a00ff',  # Indigo for water vapor
            10: '#3200aa', # Dark blue for SWIR Cirrus
            11: '#240070', # Very dark blue for SWIR
            12: '#120038'  # Almost black purple for SWIR
        }
        
        # Sample pixels from each band (sampling to avoid memory issues)
        band_samples = {i: [] for i in range(13)}
        sample_size = 10000  # Number of pixels to sample per image
        max_images = 50  # Maximum number of images to process
        
        print(f"Sampling {sample_size} pixels from up to {max_images} Sentinel-2 images...")
        
        dtype = torch.float16
        # Process a subset of images
        for path in tqdm.tqdm(sentinel2_paths[:max_images], desc="Processing Sentinel-2 images"):
            try:
                # Read the image
                img = read_tiff_to_torch(path, dtype, normalize=True)
                
                # Check if the image has 13 bands (Sentinel-2)
                if img.shape[0] != 13:
                    print(f"Skipping {path}: expected 13 bands, got {img.shape[0]}")
                    continue
                
                # Sample pixels from each band
                for band in range(13):
                    # Flatten the band and sample random pixels
                    flat_band = img[band].flatten()
                    if len(flat_band) > sample_size:
                        indices = np.random.choice(len(flat_band), sample_size, replace=False)
                        samples = flat_band[indices].tolist()
                    else:
                        samples = flat_band.tolist()
                    
                    band_samples[band].extend(samples)
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        # Create a single plot with all bands
        plt.figure(figsize=(15, 10))
        
        # Plot each band with its representative color
        for band in range(13):
            if band_samples[band]:
                sns.kdeplot(
                    band_samples[band], 
                    label=band_info[band],
                    color=band_colors[band],
                    fill=True,
                    alpha=0.2
                )
        
        plt.title("Distribution of Pixel Values Across All Sentinel-2 Bands", fontsize=16)
        plt.xlabel("Pixel Intensity", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlim([-5, 10]) 
        plt.tight_layout()
        plt.show()
        
        # Create a boxplot of all bands with the same colors
        plt.figure(figsize=(15, 8))
        boxprops = dict(linestyle='-', linewidth=1.5)
        
        # Create the boxplot with custom colors
        box_plot = plt.boxplot(
            [band_samples[i] for i in range(13)], 
            labels=[f"B{i+1}" for i in range(13)], 
            notch=True,
            patch_artist=True,  # Fill the boxes with color
            boxprops=boxprops
        )
        
        # Color the boxes according to their representative colors
        for i, box in enumerate(box_plot['boxes']):
            box.set(facecolor=band_colors[i], alpha=0.6)
            box.set(edgecolor=band_colors[i], linewidth=1.5)
        
        plt.title("Distribution of Pixel Values Across Sentinel-2 Bands", fontsize=16)
        plt.xlabel("Spectral Band", fontsize=14)
        plt.ylabel("Pixel Value", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()