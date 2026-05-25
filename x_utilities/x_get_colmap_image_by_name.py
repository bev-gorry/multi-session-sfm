import os

def get_colmap_image_by_name(images_dict, name):
    for img in images_dict.values():
        if os.path.basename(img.name) == name:
            return img
    return None