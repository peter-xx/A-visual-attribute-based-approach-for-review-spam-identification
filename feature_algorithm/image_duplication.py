from utils import get_images, to_hash


def img_hash_duplication(reviews:list):

    """根据评论图片的md5哈希值是否重复判断虚假评论"""

    pred_label = reviews.copy()
    hash_list = []

    # 将图片转换为hash值
    for review in reviews:

        imgs = get_images(review)

        for img in imgs:
            hash = to_hash(img)
            hash_list.append(hash)

    # 判断图片是否重复
    for i, hash in enumerate(hash_list):

        is_duplication = False

        for j in range(i + 1, len(hash_list)):
            if hash == hash_list[j]:

                pred_label[j // 2] = "false"
                is_duplication = True

        if is_duplication:
            pred_label[i // 2] = "false"

    return pred_label