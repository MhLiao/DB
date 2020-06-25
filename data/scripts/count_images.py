import os
from pathlib import Path


def is_image(path):
    return path.endswith(('.jpg', '.jpeg', '.bmp', '.jpeg'))


def count_images(root):
    count = 0
    for base, dirs, files in os.walk(root):
        images = [file for file in files if is_image(file)]
        count += len(images)
    return count


if __name__ == '__main__':
    _root = '../../..'

    _abs_path = str(Path(_root).absolute())
    print(f'Scanning "{_abs_path}"')
    _count = count_images(_root)
    print(f'There are {_count} images in "{_abs_path}"')
