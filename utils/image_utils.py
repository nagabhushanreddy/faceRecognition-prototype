import os
import imghdr

def is_image(path):
    """
    Check if the given file path points to an image file.
    """
    # Check if the file exists
    if not os.path.isfile(path):
        return False
    # Get the file type using imghdr
    file_type = imghdr.what(path)
    # If the file type is a known image type, return True
    if file_type in ["jpeg", "png", "gif", "bmp", "jpg"]:
        return True
    return False