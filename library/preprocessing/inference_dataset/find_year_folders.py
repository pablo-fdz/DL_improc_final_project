import os
import re

def find_year_folders(base_path):
    """Find all year folders in the inference dataset directory."""
    folders = []
    year_pattern = re.compile(r"^\d{4}$")  # Matches a 4-digit year

    if not os.path.isdir(base_path):
        print(f"Error: Base path '{base_path}' does not exist or is not a directory.")
        return folders

    print(f"Searching for year folders in: {base_path}")
    for folder_name in os.listdir(base_path):
        full_path = os.path.join(base_path, folder_name)
        if os.path.isdir(full_path) and year_pattern.match(folder_name):
            folders.append(full_path)

    print(f"Found {len(folders)} year folders.")
    if folders:
        # Sort folders by year
        folders.sort()
        print(f"First folder: {folders[0]}")
        
    return folders