def format_cls(reviews:list):

    """根据图片格式是否相同判断虚假评论"""

    pred_label = reviews.copy()

    for index, review in enumerate(reviews):

        imgs = get_images(review)

        first_img_format = Image.open(imgs[0]).format
        second_img_format = Image.open(imgs[1]).format

        if first_img_format != second_img_format:

            pred_label[index] = "false"

    return pred_label
