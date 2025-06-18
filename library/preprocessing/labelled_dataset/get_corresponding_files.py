import os
import glob

def get_corresponding_files(tiff_file):
    """Get corresponding coverage and mask files"""
    base_dir = os.path.dirname(tiff_file)
    file_name = os.path.basename(tiff_file)
    base_name, _ = os.path.splitext(file_name)
    
    # Get coverage file
    coverage_file = os.path.join(base_dir, f"{base_name}_coverage.tiff")
    if not os.path.exists(coverage_file):
        coverage_file = os.path.join(base_dir, f"{base_name}_coverage.png")
        if os.path.exists(coverage_file):
            print(f"Found PNG coverage: {coverage_file}")
        else:
            print(f"No coverage file found for {file_name}")
            coverage_file = None
    else:
        print(f"Found TIFF coverage: {coverage_file}")
    
    # Get mask file - assuming it's the same for all images in the folder
    mask_files = glob.glob(os.path.join(base_dir, "*_mask.tiff"))
    if not mask_files:
        mask_files = glob.glob(os.path.join(base_dir, "*_mask.png"))
    
    mask_file = mask_files[0] if mask_files else None
    if mask_file:
        print(f"Found mask file: {os.path.basename(mask_file)}")
    else:
        print(f"No mask file found in {base_dir}")
    
    return {
        'image': tiff_file,
        'coverage': coverage_file,
        'mask': mask_file
    }