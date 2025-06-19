import os
from glob import glob

def get_corresponding_files(img_file, file_extension='tiff'):
    """Get corresponding coverage and mask files"""
    base_dir = os.path.dirname(img_file)
    file_name = os.path.basename(img_file)
    base_name, _ = os.path.splitext(file_name)
    
    # Get coverage file
    coverage_file = os.path.join(base_dir, f"{base_name}_coverage.{file_extension}")
    if not os.path.exists(coverage_file):
        print(f"No coverage file found for {file_name}")
        coverage_file = None
    else:
        print(f"Found {file_extension} coverage: {coverage_file}")
    
    # Get mask file - assuming it's the same for all images in the folder
    mask_files = glob(os.path.join(base_dir, f"*_mask.{file_extension}"))
    if not mask_files:
        mask_files = glob(os.path.join(base_dir, f"*_{file_extension}.png"))
    
    mask_file = mask_files[0] if mask_files else None
    if mask_file:
        print(f"Found mask file: {os.path.basename(mask_file)}")
    else:
        print(f"No mask file found in {base_dir}")
    
    return {
        'image': img_file,
        'coverage': coverage_file,
        'mask': mask_file
    }