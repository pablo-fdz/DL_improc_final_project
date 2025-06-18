import os

def find_fire_event_folders(base_path):
    """Find all fire event folders in the dataset"""
    folders = []
    
    # Check if we're dealing with a single part or the parent directory
    if "Satellite_burned_area_dataset_part" in base_path:
        # Single part directory provided
        if os.path.isdir(base_path):
            print(f"Searching for fire folders in single part: {base_path}")
            # Get all subdirectories that might represent fire events
            for folder in os.listdir(base_path):
                full_path = os.path.join(base_path, folder)
                if os.path.isdir(full_path):
                    folders.append(full_path)
    else:
        # Parent directory provided - look for all parts
        print(f"Searching for dataset parts in: {base_path}")
        for part in range(1, 6):  # Dataset is divided into 5 parts
            part_path = os.path.join(base_path, f"Satellite_burned_area_dataset_part{part}")
            if os.path.exists(part_path):
                print(f"Found part {part} at {part_path}")
                for folder in os.listdir(part_path):
                    full_path = os.path.join(part_path, folder)
                    if os.path.isdir(full_path):
                        folders.append(full_path)
    
    print(f"Found {len(folders)} fire event folders")
    if len(folders) > 0:
        print(f"Example folder: {folders[0]}")
    return folders