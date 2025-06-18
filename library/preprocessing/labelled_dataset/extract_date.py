import re
from datetime import datetime

# Helper function to extract date from filenames
def extract_date(filename):
    """Extract date from filename pattern like 'sentinel1_2017-07-01.tiff'"""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d')
    return datetime.min  # Return minimum date if no match