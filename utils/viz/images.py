import cv2
import numpy as np


def generate_overview_graphic_tiles(imagepaths, targetpath):
    if len(imagepaths) == 0:
        return

    images = [cv2.imread(p) for p in imagepaths]
    blank_image = np.ones(np.shape(images[0]), np.uint8) * 255

    img_height, img_width = np.shape(images[0])[0:2]
    ratio = img_height / img_width
    items_per_row = int(np.sqrt(len(imagepaths) * ratio))

    images_rows = []

    for i, img in enumerate(images):
        if i % items_per_row == 0:
            images_rows.append([])

        if img is not None:
            images_rows[-1].append(img)
        else:
            raise Exception('File not found: {}'.format(imagepaths[i]))

    last_row = images_rows[-1]
    if len(last_row) < items_per_row:
        diff = items_per_row - len(last_row)

        for _ in range(diff):
            last_row.append(blank_image)

    def concat_tile(im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

    im_tile = concat_tile(images_rows)
    cv2.imwrite(targetpath, im_tile)


def concatenate_images(imagepaths, targetpath):
    images = [cv2.imread(p) for p in imagepaths]

    cv2.imwrite(targetpath, cv2.hconcat(images))