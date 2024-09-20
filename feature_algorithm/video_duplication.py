

def vid_hash_duplication(reviews:list):

    """根据视频的MD5哈希值是否重复判断虚假评论"""

    pred_label = reviews.copy()
    hash_list = []

    # 将存在的视频转换为hash值
    for index, review in enumerate(reviews):

        video = get_video(review)

        if len(video) > 0:
            hash = to_hash(video[0])
            hash_list.append(hash)
        else:
            hash_list.append("0")

    # 判断视频是否重复
    for i, hash in enumerate(hash_list):

        is_duplication = False

        if hash_list[i] != "0" and hash_list[i] != "false":

            for j in range(i + 1, len(hash_list)):

                if hash_list[j] != "0" and hash_list[j] != "false":

                    if hash_list[i] == hash_list[j]:
                        pred_label[j] = "false"
                        hash_list[j] = "false"
                        is_duplication = True

            if is_duplication:
                pred_label[i] = "false"
                hash_list[i] = "false"

    return pred_label