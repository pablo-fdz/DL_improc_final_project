import os

def print_directory_structure(path, max_depth=3, current_depth=0):
    """
    Prints the directory structure up to a specified depth.

    Args:
        path (str): The directory path to print.
        max_depth (int): The maximum depth to print.
        current_depth (int): The current depth in the recursion.
    """
    
    if current_depth > max_depth:
        return
    
    try:
        items = sorted(os.listdir(path))
        for item in items:
            item_path = os.path.join(path, item)
            indent = "  " * current_depth
            if os.path.isdir(item_path):
                print(f"{indent}{item}/")
                print_directory_structure(item_path, max_depth, current_depth + 1)
            else:
                print(f"{indent}{item}")
    except PermissionError:
        print(f"{indent}[Permission Denied]")