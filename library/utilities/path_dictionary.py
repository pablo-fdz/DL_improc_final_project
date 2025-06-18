from typing import Dict, List
import os

def build_path_dictionary(root_folder: str) -> Dict[str, List[str]]:
    """
    Build a dictionary with paths organized by data type.
    
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
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check if file is in coverages folder
            if 'coverages' in root.lower():
                path_dict['coverages'].append(file_path)
            
            # Check if file is in masks folder
            elif 'masks' in root.lower():
                path_dict['masks'].append(file_path)
            
            # Check if file is in images folder
            elif 'images' in root.lower():
                # Detect Sentinel-1 vs Sentinel-2 by filename
                if 'sentinel1' in file.lower() or 's1' in file.lower() or 'sar' in file.lower():
                    path_dict['sentinel1'].append(file_path)
                elif 'sentinel2' in file.lower() or 's2' in file.lower() or 'optical' in file.lower():
                    path_dict['sentinel2'].append(file_path)
                else:
                    # If no clear indicator, you might want to print the filename to see the pattern
                    print(f"Unknown image type: {file}")
    
    return path_dict