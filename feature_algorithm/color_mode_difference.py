def color_mode_cls(reviews:list):

    """根据评论中图片的颜色模式判断虚假评论"""

    pred_label = reviews.copy()

    for index, review in enumerate(reviews):

        imgs = get_images(review)

        if len(imgs) > 0:

            first_img_mode = Image.open(imgs[0]).mode
            second_img_mode = Image.open(imgs[1]).mode

        if first_img_mode != "RGB" or second_img_mode != "RGB":

            pred_label[index] = "false"

    return pred_label
