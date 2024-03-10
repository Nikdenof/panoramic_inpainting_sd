import cv2
from PIL import Image
import logging
import numpy as np
import matplotlib.pyplot as plt


def show_rgb_image(image_array):
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if len(image_array.shape) < 3:
        plt.imshow(image_array, cmap="gray")
    else:
        plt.imshow(image_array)
    plt.axis("off")
    plt.show()


def filter_mask_islands(binary_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    mask_list = []
    mask_areas = []
    for label in range(1, num_labels):
        # Create a mask for the current connected component
        island_mask = (labels == label).astype(np.uint8) * 255
        island_area = cv2.countNonZero(island_mask)
        mask_list.append(island_mask)
        mask_areas.append(island_area)

    selected_mask_index = np.argmax(mask_areas)
    selected_mask = mask_list[selected_mask_index]
    return selected_mask


def dilate_mask(mask_in, dilate_radius=8, dilate_iterations=5, verbose=False):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_radius, dilate_radius)
    )
    dilated_out = cv2.dilate(mask_in, kernel, iterations=dilate_iterations)
    mask_diff = cv2.bitwise_xor(mask_in, dilated_out)
    if verbose:
        area_diff = cv2.countNonZero(mask_diff)
        logging.info(f"Area difference before and after dilation is {area_diff}")
    return dilated_out


def mask_cv2_preprocess(mask_in, dilate_bool=True, **kwargs):
    filtered_mask = filter_mask_islands(mask_in)
    if dilate_bool:
        filtered_mask = dilate_mask(filtered_mask, **kwargs)
    return filtered_mask


def mask_pil_preprocess(mask_pil_in, **kwargs):
    mask_array = np.array(mask_pil_in)
    mask_array = mask_array.astype(np.uint8) * 255  # Convert to 0 and 255

    mask_out = mask_cv2_preprocess(mask_array, **kwargs)
    pil_out = Image.fromarray(mask_out).convert("1")
    return pil_out


def main():
    mask_test = cv2.imread("../../data/raw/test_mask.png", cv2.IMREAD_GRAYSCALE)
    show_rgb_image(mask_test)
    mask_out = mask_cv2_preprocess(mask_test, dilate_radius=5)
    mask_diff = cv2.bitwise_xor(mask_out, mask_test)
    show_rgb_image(mask_out)
    show_rgb_image(mask_diff)


if __name__ == "__main__":
    main()
