import matplotlib.pyplot as plt
from ..utilities.read_tiff_to_torch import read_tiff_to_torch
from ..utilities.read_png_to_torch import read_png_to_torch
import torch
import numpy as np

########################################
# JUST FOR INFO - MEANING OF THE BANDS IN SENTINEL-2 IMAGES
#    band_info = {
#        0: "B1 - Coastal aerosol (443 nm)",
#        1: "B2 - Blue (490 nm)",
#        2: "B3 - Green (560 nm)",
#        3: "B4 - Red (665 nm)",
#        4: "B5 - Vegetation Red Edge (705 nm)",
#        5: "B6 - Vegetation Red Edge (740 nm)",
#        6: "B7 - Vegetation Red Edge (783 nm)",
#        7: "B8 - NIR (842 nm)",
#        8: "B8A - Narrow NIR (865 nm)",
#        9: "B9 - Water vapor (945 nm)",
#        10: "B10 - SWIR Cirrus (1375 nm)",
#        11: "B11 - SWIR (1610 nm)",
#        12: "B12 - SWIR (2190 nm)"
#    }
########################################

class IndicesVisualizer:
    def __init__(self, img_path, mask_path=None):
        self.img_path = img_path
        self.mask_path = mask_path

        self.img = read_tiff_to_torch(self.img_path)
        if self.mask_path is not None:
            self.mask = read_png_to_torch(self.mask_path)
        else:
            self.mask = None

        self.NBR = (self.img[7] - self.img[11]) / (self.img[7] + self.img[11])
        self.NVDI = (self.img[7] - self.img[3]) / (self.img[7] + self.img[3])
        self.RE_NDVI = (self.img[6] - self.img[3]) / (self.img[6] + self.img[3])
        self.NDMI = (self.img[7] - self.img[10]) / (self.img[7] + self.img[10])

    def visualize_indices(self, figsize=[9, 9]):
        """
        Plots image alongside flood mask and water mask, respectively. 
        """

        indices = {
            'NBR': self.NBR,
            'NVDI': self.NVDI,
            'RE-NDVI': self.RE_NDVI,
            'NDMI': self.NDMI
        }

        fig, axes = plt.subplots(3, 2, figsize=figsize)
        axes = axes.flatten()

        # Plot RGB image (bands 4,3,2)
        rgb = self.img[[3, 2, 1]].permute(1, 2, 0).numpy()
        rgb = rgb * 3
        axes[0].imshow(rgb)
        axes[0].set_title("RGB (B4,B3,B2)")
        axes[0].axis('off')

        # Plot mask or white image if mask is missing
        if self.mask is not None:
            axes[1].imshow(self.mask.squeeze().numpy(), cmap='gray')
            axes[1].set_title("Mask")
        else:
            # Draw a white image in the mask spot
            h, w = self.img.shape[1], self.img.shape[2]
            axes[1].imshow(np.ones((h, w)), cmap='gray', vmin=0, vmax=1)
            axes[1].set_title("Mask (missing)")
        axes[1].axis('off')

        # Plot indices
        for i, (name, index) in enumerate(indices.items()):
            axes[i+2].imshow(index.numpy(), cmap='plasma', vmin=0, vmax=1)
            axes[i+2].set_title(name)
            axes[i+2].axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_correlations(self, size=[5,5]):
        """
        Visualizes the correlation between the indices and the mask values.
        """
        import seaborn as sns
        import pandas as pd

        indices = {
            'NBR': self.NBR,
            'NVDI': self.NVDI,
            'RE-NDVI': self.RE_NDVI,
            'NDMI': self.NDMI
        }

        if self.mask is None:
            print("No mask provided, cannot compute correlations.")
            return

        # Create a DataFrame for the indices and mask
        data = pd.DataFrame({name: idx.flatten().numpy() for name, idx in indices.items()
                             } | {'Mask': self.mask.squeeze().flatten().numpy()})
        # Calculate correlation matrix
        corr = data.corr()

        # Plot the heatmap
        plt.figure(figsize=size)
        sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title("Correlation between Indices and Mask")
        plt.show()