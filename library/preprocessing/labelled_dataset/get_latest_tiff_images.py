from glob import glob
import os

def get_latest_tiff_images(folder_path):
    """Get the latest dated .tiff images from the folder for each Sentinel type"""
    sentinel1_files = glob.glob(os.path.join(folder_path, "sentinel1_*.tiff"))
    sentinel2_files = glob.glob(os.path.join(folder_path, "sentinel2_*.tiff"))
    
    print(f"Found {len(sentinel1_files)} Sentinel-1 files and {len(sentinel2_files)} Sentinel-2 files in {folder_path}")
    
    # Filter out coverage files
    sentinel1_files = [f for f in sentinel1_files if "_coverage" not in f]
    sentinel2_files = [f for f in sentinel2_files if "_coverage" not in f]
    
    print(f"After filtering coverage files: {len(sentinel1_files)} S1 and {len(sentinel2_files)} S2 files")
    
    # Sort by date, newest first
    sentinel1_files.sort(reverse=True)  # Sort in descending order to get the most recent files first
    sentinel2_files.sort(reverse=True)  # Sort in descending order to get the most recent files first
    
    latest_files = []
    
    # Get the latest file for each type if available
    if sentinel1_files:
        latest_files.extend(sentinel1_files[:min(1, len(sentinel1_files))])
    if sentinel2_files:
        latest_files.extend(sentinel2_files[:min(1, len(sentinel2_files))])
    
    print(f"Selected {len(latest_files)} latest files: {[os.path.basename(f) for f in latest_files]}")
    return latest_files