import os
import re
from collections import defaultdict

def get_matching_images(year_folder_path, file_extension='tiff'):
    """
    Groups Sentinel-1 and Sentinel-2 images by their common suffix (date and coordinates).
    If an image is missing for a specific type (Sentinel-1 or Sentinel-2),
    it will still be included in the output with a None value for that type.
    This function assumes that the images are stored in a subfolder named 'images'
    within the specified year folder path.

    Args:
        year_folder_path (str): Path to a specific year's folder (e.g., '.../catalunya_fire_imgs/2015').
        file_extension (str): The file extension of the images to look for.

    Returns:
        list: A list of dictionaries. Each dictionary represents a unique fire event (by suffix)
              and contains paths to the corresponding Sentinel-1 and/or Sentinel-2 images.
              Example: [{'id': '...', 'sentinel1': 'path/to/s1', 'sentinel2': 'path/to/s2'}]
    """
    images_path = os.path.join(year_folder_path, 'images')
    if not os.path.isdir(images_path):
        print(f"Warning: 'images' subfolder not found in '{year_folder_path}'")
        return []

    # Regex to capture satellite type (sentinel1 or sentinel2) and the common identifier
    # e.g., sentinel1_2015-07-13_lon3-161_lat42-432.tiff
    pattern = re.compile(rf"^(sentinel[12])_(.+)\.{file_extension}$")
    
    fire_events = defaultdict(dict)

    for filename in os.listdir(images_path):
        match = pattern.match(filename)
        if match:
            satellite_type = match.group(1)  # 'sentinel1' or 'sentinel2'
            common_suffix = match.group(2)   # '2015-07-13_lon3-161_lat42-432'
            
            # Store the full path under the satellite type for that common suffix
            fire_events[common_suffix][satellite_type] = os.path.join(images_path, filename)

    # Convert the grouped dictionary to a list of dictionaries
    matched_list = []
    for suffix, paths in sorted(fire_events.items()):
        event_data = {
            'id': suffix,
            'sentinel1': paths.get('sentinel1', None),  # Set to None if not found
            'sentinel2': paths.get('sentinel2', None)  # Set to None if not found
        }
        matched_list.append(event_data)
        
    print(f"Found {len(matched_list)} unique fire events in {os.path.basename(year_folder_path)}")
    return matched_list