from utils import get_images
from PIL import Image


def all_diff_resolution(reviews:list):

    """根据评论中图片的宽度高度是否都不相同判断虚假评论"""

    pred_label = reviews.copy()

    for index, review in enumerate(reviews):

        images = get_images(review)

        first_img_resolution = Image.open(images[0]).size
        second_img_resolution = Image.open(images[1]).size

        if first_img_resolution[0] != second_img_resolution[0] and first_img_resolution[1] != second_img_resolution[1]:
            pred_label[index] = "false"

    return pred_label


def too_lg_resolution(reviews:list, threshold_resolution:list=[324 * 1.91, 425 * 1.91]):

    """"通过过大的分辨率判断虚假评论"""

    pred_label = reviews.copy()

    for index, review in enumerate(reviews):

        images = get_images(review)

        first_img_resolution = Image.open(images[0]).size
        second_img_resolution = Image.open(images[1]).size

        if (first_img_resolution[0] > threshold_resolution[0] or first_img_resolution[1] > threshold_resolution[1]
                or second_img_resolution[0] > threshold_resolution[0] or second_img_resolution[1] > threshold_resolution[1]):
            pred_label[index] = "false"

    return pred_label


def same_resolution(reviews:list):

    """根据评论中图片的宽度高度是否都相同判断虚假评论"""

    pred_label = reviews.copy()

    for index, review in enumerate(reviews):

        if review != "finished":

            images = get_images(review)

            first_img_resolution = Image.open(images[0]).size
            second_img_resolution = Image.open(images[1]).size

            if first_img_resolution[0] == second_img_resolution[0] and first_img_resolution[1] == second_img_resolution[1]:
                # pred_label[index] = "true"
                continue
            else:
                pred_label[index] = "false"
        else:
            pred_label[index] = "false"

    return pred_label