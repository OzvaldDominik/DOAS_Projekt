import argparse
import os
import numpy as np
import cv2
import random

image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
GRID_ROWS = 16
GRID_COLS = 16


def get_neighbors(r, c,grid_rows=16, grid_cols=16):
    neighbors = []
    for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + i
        nc = c + j
        if 0 <= nr <= (grid_rows - 1):
            neighbors.append((nr, nc))
    return neighbors


def generate_mask(img, grid_rows=16, grid_cols=16, limit=(GRID_COLS // 2) * (GRID_ROWS // 2),neighbors=False):
    height, width = img.shape[:2]
    cell_height = height // grid_rows
    cell_width = width // grid_cols
    gird = np.zeros((grid_rows, grid_cols))
    middle_cols = list(range((grid_cols // 2) - (grid_cols // 4), (grid_cols // 2) + (grid_cols // 4)))
    middle_rows = list(range((grid_rows // 2) - (grid_rows // 4), (grid_rows // 2) + (grid_rows // 2)))

    candidates = [(r, c) for r in middle_rows for c in middle_cols]
    random.shuffle(candidates)
    for r, c in candidates[:limit]:
        if (not any(gird[nr, nc] for nr, nc in get_neighbors(r, c,grid_rows,grid_cols))) or neighbors:
            gird[r, c] = 1

    mask = np.ones(img.shape[:2])

    for i in range(grid_rows):
        for j in range(grid_cols):
            if gird[i, j]:
                y1 = i * cell_height
                y2 = (i + 1) * cell_height
                x1 = j * cell_width
                x2 = (j + 1) * cell_width
                mask[y1:y2, x1:x2] = 0

    return np.expand_dims(mask, axis=-1)


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(prog='MaskGenerator')
    parser.add_argument('-s', '--source_folder', default=".", required=False)
    parser.add_argument('-d', '--dest_folder', default="out", required=False)
    parser.add_argument('-l', '--limit', default=(GRID_COLS // 2) * (GRID_ROWS // 2), required=False, type=int)
    parser.add_argument('-g', '--grid_size', default=0, required=False, type=int)
    parser.add_argument('-m','--multi', action='store_true')
    parser.add_argument('-n', '--neighbors', action='store_true')

    args = parser.parse_args()
    source_folder = args.source_folder
    dest_folder = args.dest_folder
    limit = args.limit
    grid_size = args.grid_size
    multi = args.multi
    neighbors = args.neighbors
    if grid_size < 0:
        grid_size = 0
    elif grid_size > 12:
        grid_size = 12

    if not source_folder.endswith('/'):
        source_folder += '/'
    if not dest_folder.endswith('/'):
        dest_folder += '/'

    image_names = [name for name in os.listdir(args.source_folder) if name.lower().endswith(image_extensions)]

    for image in image_names:
        img = cv2.imread(source_folder + image)
        mask = np.ones((img.shape[0], img.shape[1], 1))
        if multi :
            for i in range(0,12,4):
                mask = mask * generate_mask(img, grid_rows=GRID_ROWS - i, grid_cols=GRID_COLS - i, limit=32//(i+1),neighbors=neighbors)
        else:
            mask = generate_mask(img, grid_rows=GRID_ROWS - grid_size, grid_cols=GRID_COLS - grid_size, limit=limit,neighbors=neighbors)

        img = img * mask
        print(f"Storing: {dest_folder + image} ")
        cv2.imwrite(dest_folder + image, img)
