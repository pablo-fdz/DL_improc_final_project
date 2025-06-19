from typing import Dict, List
import os


def build_path_dictionary(root_folder: str) -> Dict[str, List[str]]:
    """
    Build a dictionary with paths organized by data type.
    Works with both tiled dataset and original satellite dataset structures.
    
    Args:
        root_folder: Path to the root folder containing the dataset
        
    Returns:
        Dictionary with keys: 'coverages', 'masks', 'sentinel1', 'sentinel2'
        Each key contains a list of file paths for that data type
    """
    path_dict = {
        'coverages': [],
        'masks': [], 
        'sentinel1': [],
        'sentinel2': []
    }
    
    for root, _, files in os.walk(root_folder):
        for file in files:
            # Skip non-image files
            if not (file.lower().endswith('.tiff') or file.lower().endswith('.png')):
                continue
                
            file_path = os.path.join(root, file)
            
            # Check if file is in coverages folder (tiled dataset structure)
            if 'coverages' in root.lower():
                path_dict['coverages'].append(file_path)
            
            # Check if file is in masks folder (tiled dataset structure)  
            elif 'masks' in root.lower():
                path_dict['masks'].append(file_path)
            
            # Check if file is in images folder (tiled dataset structure)
            elif 'images' in root.lower():
                # Detect Sentinel-1 vs Sentinel-2 by filename
                if 'sentinel1' in file.lower() or 's1' in file.lower() or 'sar' in file.lower():
                    path_dict['sentinel1'].append(file_path)
                elif 'sentinel2' in file.lower() or 's2' in file.lower() or 'optical' in file.lower():
                    path_dict['sentinel2'].append(file_path)
                else:
                    print(f"Unknown image type: {file}")
            
            # Handle original satellite dataset structure (files directly in project folders)
            else:
                # Check for mask files (PNG files ending with _mask)
                if file.lower().endswith('_mask.png'):
                    path_dict['masks'].append(file_path)
                
                # Check for sentinel1 files
                elif 'sentinel1' in file.lower() and file.lower().endswith('.tiff'):
                    path_dict['sentinel1'].append(file_path)
                
                # Check for sentinel2 files  
                elif 'sentinel2' in file.lower() and file.lower().endswith('.tiff'):
                    path_dict['sentinel2'].append(file_path)
                
                # Check for coverage files (remaining TIFF files that aren't sentinel)
                elif (file.lower().endswith('.tiff') and 
                      'sentinel' not in file.lower() and 
                      '_mask' not in file.lower()):
                    path_dict['coverages'].append(file_path)
    
    return path_dict