import argparse
import os
import numpy as np
import cv2
import random

image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
GRID_ROWS = 16
GRID_COLS = 16

def get_neighbors(r, c):
    neighbors = []
    for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + i
        nc = c + j
        if 0 <= nr <= (GRID_ROWS-1):
            neighbors.append((nr,nc))
    return neighbors

def generate_mask(img,limit=GRID_COLS*(GRID_ROWS//2)):
    height, width = img.shape[:2]
    cell_height = height // GRID_ROWS
    cell_width = width // GRID_COLS
    gird = np.zeros((GRID_ROWS, GRID_COLS))
    middle_cols = list(range((GRID_COLS//2)-(GRID_ROWS//4), (GRID_COLS//2)+(GRID_ROWS//4)))

    candidates = [(r, c) for r in range(GRID_ROWS) for c in middle_cols]
    random.shuffle(candidates)
    for r,c in candidates[:limit]:
        if not any(gird[nr,nc] for nr , nc in get_neighbors(r,c)):
            gird[r,c]=1

    mask = np.ones(img.shape[:2])

    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            if gird[i,j]:
                y1 = i*cell_height
                y2 = (i+1)*cell_height
                x1 = j*cell_width
                x2 = (j+1)*cell_width
                mask[y1:y2,x1:x2]=0

    return np.expand_dims(mask, axis=-1)


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(prog='MaskGenerator')
    parser.add_argument('-s', '--source_folder', default=".", required=False)
    parser.add_argument('-d', '--dest_folder', default="out", required=False)
    parser.add_argument('-l', '--limit', default=GRID_COLS*(GRID_ROWS//2), required=False,type=int)

    args = parser.parse_args()
    source_folder = args.source_folder
    dest_folder = args.dest_folder
    limit = args.limit

    if not source_folder.endswith('/'):
        source_folder += '/'
    if not dest_folder.endswith('/'):
        dest_folder += '/'

    image_names = [name for name in os.listdir(args.source_folder) if name.lower().endswith(image_extensions)]

    for image in image_names:
        img = cv2.imread(source_folder + image)
        mask = generate_mask(img,limit)
        img = img*mask
        print(f"Storing: {dest_folder+image} ")
        cv2.imwrite(dest_folder+image,img)



