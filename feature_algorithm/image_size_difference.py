

def size_cls(reviews:list, threshold:float=0.649):
    # 余下评论0.3
    # def size_cls(reviews:list, threshold:float=0.3):

    """根据评论中图片间的大小差异判断虚假评论"""

    pred_label = reviews.copy()
    size_ratio = []

    max_img_size, min_img_size = get_image_size(reviews)

    for index, size in enumerate(min_img_size):

        if size != "finished":
            ratio = size / max_img_size[index]
            size_ratio.append(ratio)
        else:
            size_ratio.append("finished")

    for index, img_size_ratio in enumerate(size_ratio):

        if img_size_ratio != "finished":
            if img_size_ratio < threshold:
                pred_label[index] = "false"
        else:
            pred_label[index] = "false"

    return pred_label