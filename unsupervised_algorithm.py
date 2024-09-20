from utils import *
import cv2
from sklearn.model_selection import KFold
from feature_algorithm.image_duplication import img_hash_duplication
from feature_algorithm.video_duplication import vid_hash_duplication
from feature_algorithm.resolution_difference import *
from feature_algorithm.color_mode_difference import color_mode_cls
from feature_algorithm.image_size_difference import size_cls
from feature_algorithm.format_difference import format_cls


def k_fold_cross_validation(data:list=None,
                            label:list=None,
                            k:int=5,
                            shuffle:bool=False,
                            random_state=None,
                            method="top8"):

    """k折交叉验证，返回平均训练精度和平均测试精度"""

    folder = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    mean_acc = 0
    mean_precision = 0
    mean_recall = 0
    mean_f1 = 0
    mean_f05 = 0
    mean_f2 = 0
    mean_auc_area = 0

    for train_index, test_index in folder.split(data, label):

        # train_data = extract_elements(data, train_index)
        # train_label = extract_elements(label, train_index)

        test_data = extract_elements(data, test_index)
        test_label = extract_elements(label, test_index)

        if method == "top8":
            test_pred = top8(test_data)
        elif method == "test":
            test_pred = test_method(test_data)

        acc = accuracy(test_label, test_pred)
        precision, recall, f1 = model_metrics(test_label, test_pred)
        f05 = (1 + 0.5 ** 2) * precision * recall / (0.5 ** 2 * precision + recall)
        f2 = (1 + 2 ** 2) * precision * recall / (2 ** 2 * precision + recall)
        auc_area = draw_roc(test_label, test_pred, 1, False)

        mean_acc += acc
        mean_precision += precision
        mean_recall += recall
        mean_f1 += f1
        mean_f05 += f05
        mean_f2 += f2
        mean_auc_area += auc_area

    mean_acc = mean_acc / k
    mean_precision = mean_precision / k
    mean_recall = mean_recall / k
    mean_f1 = mean_f1 / k
    mean_f05 = mean_f05 / k
    mean_f2 = mean_f2 / k
    mean_auc_area = mean_auc_area / k

    return (mean_acc,
            mean_precision,
            mean_recall,
            mean_f1,
            mean_f05,
            mean_f2,
            mean_auc_area)


def top8_plus(reviews:list):

    """运用8个特征对评论进行分类（叠加法）"""

    ihd_pred_label = img_hash_duplication(reviews)
    pred_label = ihd_pred_label.copy()
    vhd_pred_label = vid_hash_duplication(reviews)
    adr_pred_label = all_diff_resolution(reviews)
    cmc_pred_label = color_mode_cls(reviews)
    tlr_pred_label = too_lg_resolution(reviews)
    fc_pred_label = format_cls(reviews)
    sr_pred_label = same_resolution(reviews)
    sc_pred_label = size_cls(reviews)

    for index, label in enumerate(pred_label):

        if label != vhd_pred_label[index]:

            pred_label[index] = "false"

    for index, label in enumerate(pred_label):

        if label != adr_pred_label[index]:

            pred_label[index] = "false"

    for index, label in enumerate(pred_label):

        if label != cmc_pred_label[index]:

            pred_label[index] = "false"

    for index, label in enumerate(pred_label):

        if label != tlr_pred_label[index]:

            pred_label[index] = "false"

    for index, label in enumerate(pred_label):

        if label != fc_pred_label[index]:

            pred_label[index] = "false"

    for index, label in enumerate(pred_label):

        if label != sr_pred_label[index]:

            pred_label[index] = "false"

    for index, label in enumerate(pred_label):

        if label != sc_pred_label[index]:

            pred_label[index] = "false"

    for i in range(len(pred_label)):
        if pred_label[i] == "false":
            pred_label[i] = 1
        else:
            pred_label[i] = 0

    return pred_label


def top8_vote(reviews:list):

    """运用前8个特征对评论进行分类（硬投票法）"""

    pred_label = reviews.copy()
    ihd_pred_label = img_hash_duplication(reviews)
    vhd_pred_label = vid_hash_duplication(reviews)
    adr_pred_label = all_diff_resolution(reviews)
    cmc_pred_label = color_mode_cls(reviews)
    tlr_pred_label = too_lg_resolution(reviews)
    fc_pred_label = format_cls(reviews)
    sr_pred_label = same_resolution(reviews)
    sc_pred_label = size_cls(reviews)

    result_list = [ihd_pred_label, vhd_pred_label, adr_pred_label, cmc_pred_label, tlr_pred_label, fc_pred_label, sr_pred_label, sc_pred_label]

    for i, _ in enumerate(pred_label):

        true_count = 0
        false_count = 0

        for result in result_list:

            if result[i] != "false":
                true_count += 1
            else:
                false_count += 1

        if false_count > true_count:
            pred_label[i] = "false"

    for i in range(len(pred_label)):
        if pred_label[i] == "false":
            pred_label[i] = 1
        else:
            pred_label[i] = 0

    return pred_label


# 平均法
def top8_averaging(reviews:list):

    """运用前8个特征对评论进行分类（平均法）"""

    pred_label = []

    ihd_pred_label = img_hash_duplication(reviews)
    vhd_pred_label = vid_hash_duplication(reviews)
    adr_pred_label = all_diff_resolution(reviews)
    cmc_pred_label = color_mode_cls(reviews)
    tlr_pred_label = too_lg_resolution(reviews)
    fc_pred_label = format_cls(reviews)
    sr_pred_label = same_resolution(reviews)
    sc_pred_label = size_cls(reviews)

    ihd_pro = []
    vhd_pro = []
    adr_pro = []
    cmc_pro = []
    tlr_pro = []
    fc_pro = []
    sr_pro = []
    sc_pro = []

    ihd_weight = 1.0
    vhd_weight = 1.0
    adr_weight = 1.0
    cmc_weight = 1.0
    tlr_weight = 1.0
    fc_weight = 1.0
    sr_weight = 1.0
    sc_weight = 1.0

    ihd_weight_false = 1 + 354 / 2628
    vhd_weight_false = 1 + 254 / 2628
    adr_weight_false = 1 + 338 / 2628
    cmc_weight_false = 1 + 179 / 2628
    tlr_weight_false = 1 + 584 / 2628
    fc_weight_false = 1 + 83 / 2628
    sr_weight_false = 1 + 505 / 2628
    sc_weight_false = 1 + 331 / 2628

    ihd_weight_true = 1 + 19 / 366
    vhd_weight_true = 1 + 4 / 366
    adr_weight_true = 1 + 7 / 366
    cmc_weight_true = 1 + 1 / 366
    tlr_weight_true = 1 + 5 / 366
    fc_weight_true = 1 + 1 / 366
    sr_weight_true = 1 + 180 / 366
    sc_weight_true = 1 + 149 / 366

    for i, _ in enumerate(reviews):
        if ihd_pred_label[i] == "false":
            # ihd_pro.append([584/589 * ihd_weight, 5/589 * ihd_weight])
            ihd_pro.append([584/589 * ihd_weight_false, 5/589 * ihd_weight_true])
        else:
            ihd_pro.append([72/337, 265/337])

        if vhd_pred_label[i] == "false":
            # vhd_pro.append([354/373 * vhd_weight, 19/373 * vhd_weight])
            vhd_pro.append([354/373 * vhd_weight_false, 19/373 * vhd_weight_true])
        else:
            vhd_pro.append([446/1227, 781/1227])

        if adr_pred_label[i] == "false":
            # adr_pro.append([338/345 * adr_weight, 7/345 * adr_weight])
            adr_pro.append([338/345 * adr_weight_false, 7/345 * adr_weight_true])
        else:
            adr_pro.append([462/1255, 793/1255])

        if cmc_pred_label[i] == "false":
            # cmc_pro.append([179/180 * cmc_weight, 1/180 * cmc_weight])
            cmc_pro.append([179/180 * cmc_weight_false, 1/180 * cmc_weight_true])
        else:
            cmc_pro.append([621/1420, 799/1420])

        if tlr_pred_label[i] == "false":
            # tlr_pro.append([254/258 * tlr_weight, 4/258 * tlr_weight])
            tlr_pro.append([254/258 * tlr_weight_false, 4/258 * tlr_weight_true])
        else:
            tlr_pro.append([91/257, 166/257])

        if fc_pred_label[i] == "false":
            # fc_pro.append([83/84 * fc_weight, 1/84 * fc_weight])
            fc_pro.append([83/84 * fc_weight_false, 1/84 * fc_weight_true])
        else:
            fc_pro.append([715/1516, 799/1516])

        if sr_pred_label[i] == "false":
            # sr_pro.append([101/137 * sr_weight, 36/137 * sr_weight])
            sr_pro.append([101/137 * sr_weight_false, 36/137 * sr_weight_true])
        else:
            sr_pro.append([59/183, 124/183])

        if sc_pred_label[i] == "false":
            # sc_pro.append([331/480 * sc_weight, 149/480 * sc_weight])
            sc_pro.append([331/480 * sc_weight_false, 149/480 * sc_weight_true])
        else:
            sc_pro.append([469/1120, 651/1120])

    for i, _ in enumerate(reviews):
        temp = []
        # 算术平均
        # pro = (ihd_pro[i][0] + vhd_pro[i][0] + adr_pro[i][0] + cmc_pro[i][0] + tlr_pro[i][0] + fc_pro[i][0] + sr_pro[i][0] + sc_pro[i][0]) / 8
        # 几何平均
        # pro = (ihd_pro[i][0] * vhd_pro[i][0] * adr_pro[i][0] * cmc_pro[i][0] * tlr_pro[i][0] * fc_pro[i][0] * sr_pro[i][0] * sc_pro[i][0]) ** (1 / 8)
        # 平方平均
        # pro = ((ihd_pro[i][0] ** 2 + vhd_pro[i][0] ** 2 + adr_pro[i][0] ** 2 + cmc_pro[i][0] ** 2 + tlr_pro[i][0] ** 2 + fc_pro[i][0] ** 2 + sr_pro[i][0] ** 2 + sc_pro[i][0] ** 2) / 5) ** (1 / 2)
        # 调和平均
        # pro = 8 / (1 / ihd_pro[i][0] + 1 / vhd_pro[i][0] + 1 / adr_pro[i][0] + 1 / cmc_pro[i][0] + 1 / tlr_pro[i][0] + 1 / fc_pro[i][0] + 1 / sr_pro[i][0] + 1 / sc_pro[i][0])
        # 加权平均
        pro = (ihd_pro[i][0] + vhd_pro[i][0] + adr_pro[i][0] + cmc_pro[i][0] + tlr_pro[i][0] + fc_pro[i][0] + sr_pro[i][0] + sc_pro[i][0])
        temp.append(pro)

        # pro = (ihd_pro[i][1] + vhd_pro[i][1] + adr_pro[i][1] + cmc_pro[i][1] + tlr_pro[i][1] + fc_pro[i][1] + sr_pro[i][1] + sc_pro[i][1]) / 8
        # pro = (ihd_pro[i][1] * vhd_pro[i][1] * adr_pro[i][1] * cmc_pro[i][1] * tlr_pro[i][1] * fc_pro[i][1] * sr_pro[i][1] * sc_pro[i][1]) ** (1 / 8)
        # pro = ((ihd_pro[i][1] ** 2 + vhd_pro[i][1] ** 2 + adr_pro[i][1] ** 2 + cmc_pro[i][1] ** 2 + tlr_pro[i][1] ** 2 + fc_pro[i][1] ** 2 + sr_pro[i][1] ** 2 + sc_pro[i][1] ** 2) / 5) ** (1 / 2)
        # pro = 8 / (1 / ihd_pro[i][1] + 1 / vhd_pro[i][1] + 1 / adr_pro[i][1] + 1 / cmc_pro[i][1] + 1 / tlr_pro[i][1] + 1 / fc_pro[i][1] + 1 / sr_pro[i][1] + 1 / sc_pro[i][1])
        pro = (ihd_pro[i][1] + vhd_pro[i][1] + adr_pro[i][1] + cmc_pro[i][1] + tlr_pro[i][1] + fc_pro[i][1] + sr_pro[i][1] + sc_pro[i][1])
        temp.append(pro)
        pred_label.append(temp)

    for i, pro in enumerate(pred_label):
        if pro[0] > pro[1]:
            pred_label[i] = 1
        else:
            pred_label[i] = 0

    return pred_label


def top8(reviews: list):

    """
    叠加法+平均法
    :param reviews:
    :return:
    """

    ihd_pro = []
    vhd_pro = []
    adr_pro = []
    cmc_pro = []
    tlr_pro = []
    fc_pro = []
    sr_pro = []
    sc_pro = []

    ihd_weight = 1 + 354 / 2628
    vhd_weight = 1 + 254 / 2628
    adr_weight = 1 + 338 / 2628
    cmc_weight = 1 + 179 / 2628
    tlr_weight = 1 + 584 / 2628
    fc_weight = 1 + 83 / 2628
    sr_weight = 1 + 505 / 2628
    sc_weight = 1 + 331 / 2628
    #
    sr_weight_true = 1 + 180 / 366
    sc_weight_true = 1 + 149 / 366

    ihd_pred_label = img_hash_duplication(reviews)
    vhd_pred_label = vid_hash_duplication(reviews)
    adr_pred_label = all_diff_resolution(reviews)
    tlr_pred_label = too_lg_resolution(reviews)
    cmc_pred_label = color_mode_cls(reviews)
    fc_pred_label = format_cls(reviews)

    pred_label = ihd_pred_label.copy()

    for index, label in enumerate(pred_label):

        if label != vhd_pred_label[index]:
            pred_label[index] = "false"

    for index, label in enumerate(pred_label):

        if label != adr_pred_label[index]:
            pred_label[index] = "false"

    for index, label in enumerate(pred_label):

        if label != cmc_pred_label[index]:
            pred_label[index] = "false"

    for index, label in enumerate(pred_label):

        if label != tlr_pred_label[index]:
            pred_label[index] = "false"

    for index, label in enumerate(pred_label):

        if label != fc_pred_label[index]:
            pred_label[index] = "false"

    for i, review in enumerate(reviews):

        if pred_label[i] == "false":
            reviews[i] = "finished"
            ihd_pro.append([1, 0])
            vhd_pro.append([1, 0])
            adr_pro.append([1, 0])
            cmc_pro.append([1, 0])
            tlr_pro.append([1, 0])
            fc_pro.append([1, 0])
        else:
            ihd_pro.append([446 / 1227, 781 / 1227])
            vhd_pro.append([621 / 1420, 799 / 1420])
            adr_pro.append([462 / 1255, 793 / 1255])
            cmc_pro.append([91 / 257, 166 / 257])
            tlr_pro.append([72 / 337, 265 / 337])
            fc_pro.append([715 / 1516, 799 / 1516])

    sc_pred_label = size_cls(reviews)
    sr_pred_label = same_resolution(reviews)

    for i in range(0, len(sc_pred_label)):
        if sc_pred_label[i] == "false":
            # 原概率
            # sc_pro.append([331 / 480, 149 / 480])
            sc_pro.append([331 / 480 * sc_weight, 149 / 480 * sc_weight_true])
        else:
            sc_pro.append([469 / 1120, 651 / 1120])

        if sr_pred_label[i] == "false":
            # sr_pro.append([101 / 137, 36 / 137])
            sr_pro.append([101 / 137 * sr_weight, 36 / 137 * sr_weight_true])
        else:
            sr_pro.append([59 / 183, 124 / 183])

    pred_label = []
    for i in range(0, len(reviews)):
        temp = []
        # 算术平均
        # pro = (ihd_pro[i][0] + vhd_pro[i][0] + adr_pro[i][0] + cmc_pro[i][0] + tlr_pro[i][0] + fc_pro[i][0] + sc_pro[i][0] + sr_pro[i][0]) / 8
        # 几何平均
        # pro = (ihd_pro[i][0] * vhd_pro[i][0] * adr_pro[i][0] * cmc_pro[i][0] * tlr_pro[i][0] * fc_pro[i][0] * sc_pro[i][0] * sr_pro[i][0]) ** (1 / 8)
        # 平方平均
        # pro = ((ihd_pro[i][0] ** 2 + vhd_pro[i][0] ** 2 + adr_pro[i][0] ** 2 + cmc_pro[i][0] ** 2 + tlr_pro[i][0] ** 2 + fc_pro[i][0] ** 2 + sr_pro[i][0] ** 2 + sc_pro[i][0] ** 2) / 8) ** (1 / 2)
        # 调和平均
        # pro = 8 / (1 / ihd_pro[i][0] + 1 / vhd_pro[i][0] + 1 / adr_pro[i][0] + 1 / cmc_pro[i][0] + 1 / tlr_pro[i][0] + 1 / fc_pro[i][0] + 1 / sc_pro[i][0] + 1 / sr_pro[i][0])
        # pro = 6 / (1 / ihd_pro[i][0] + 1 / vhd_pro[i][0] + 1 / adr_pro[i][0] + 1 / fc_pro[i][0] + 1 / sc_pro[i][0] + 1 / sr_pro[i][0])
        # 加权平均
        pro = (ihd_pro[i][0] + vhd_pro[i][0] + adr_pro[i][0] + cmc_pro[i][0] + tlr_pro[i][0] + fc_pro[i][0] + sr_pro[i][0] + sc_pro[i][0])
        temp.append(pro)

        # pro = (ihd_pro[i][1] + vhd_pro[i][1] + adr_pro[i][1] + cmc_pro[i][1] + tlr_pro[i][1] + fc_pro[i][1] + sc_pro[i][1] + sr_pro[i][1]) / 8
        # pro = (ihd_pro[i][1] * vhd_pro[i][1] * adr_pro[i][1] * cmc_pro[i][1] * tlr_pro[i][1] * fc_pro[i][1] * sc_pro[i][1] * sr_pro[i][1]) ** (1 / 8)
        # pro = ((ihd_pro[i][1] ** 2 + vhd_pro[i][1] ** 2 + adr_pro[i][1] ** 2 + cmc_pro[i][1] ** 2 + tlr_pro[i][1] ** 2 + fc_pro[i][1] ** 2 + sr_pro[i][1] ** 2 + sc_pro[i][1] ** 2) / 8) ** (1 / 2)
        # if ihd_pro[i][1] == 0:
            # pro = 8 / (1 / 0.01 + 1 / 0.01 + 1 / 0.01 + 1 / 0.01 + 1 / 0.01 + 1 / 0.01 + 1 / sc_pro[i][1] + 1 / sr_pro[i][1])
            # pro = 6 / (1 / 0.01 + 1 / 0.01 + 1 / 0.01 + 1 / 0.01 + 1 / sc_pro[i][1] + 1 / sr_pro[i][1])
        # else:
            # pro = 8 / (1 / ihd_pro[i][1] + 1 / vhd_pro[i][1] + 1 / adr_pro[i][1] + 1 / cmc_pro[i][1] + 1 / tlr_pro[i][1] + 1 / fc_pro[i][1] + 1 / sc_pro[i][1] + 1 / sr_pro[i][1])
            # pro = 6 / (1 / ihd_pro[i][1] + 1 / vhd_pro[i][1] + 1 / adr_pro[i][1] + 1 / fc_pro[i][1] + 1 / sc_pro[i][1] + 1 / sr_pro[i][1])
        pro = (ihd_pro[i][1] + vhd_pro[i][1] + adr_pro[i][1] + cmc_pro[i][1] + tlr_pro[i][1] + fc_pro[i][1] + sr_pro[i][1] + sc_pro[i][1])
        temp.append(pro)

        pred_label.append(temp)

    for i, pro in enumerate(pred_label):
        if pro[0] > pro[1]:
            pred_label[i] = 1
        else:
            pred_label[i] = 0

    return pred_label

# def top8(reviews: list):
#
#     ihd_weight = 1 + 354 / 2044
#     vhd_weight = 1 + 254 / 2044
#     adr_weight = 1 + 338 / 2044
#     cmc_weight = 1 + 179 / 2044
#     # tlr_weight = 1 + 584 / 2628
#     fc_weight = 1 + 83 / 2044
#     sr_weight = 1 + 505 / 2044
#     sc_weight = 1 + 331 / 2044
#     sr_weight_true = 1 + 180 / 329
#     sc_weight_true = 1 + 149 / 329
#
#     ihd_pred_label = img_hash_duplication(reviews)
#     vhd_pred_label = vid_hash_duplication(reviews)
#     adr_pred_label = all_diff_resolution(reviews)
#     # tlr_pred_label = too_lg_resolution(reviews)
#     cmc_pred_label = color_mode_cls(reviews)
#     fc_pred_label = format_cls(reviews)
#     for i in range(0, len(reviews)):
#         if (ihd_pred_label[i] == "false"
#                 or vhd_pred_label[i] == "false"
#                 or adr_pred_label[i] == "false"
#                 # or tlr_pred_label[i] == "false"
#                 or cmc_pred_label[i] == "false"
#                 or fc_pred_label[i] == "false"):
#             reviews[i] = "finished"
#     sc_pred_label = size_cls(reviews)
#     sr_pred_label = same_resolution(reviews)
#
#     for i in range(0, len(reviews)):
#         # 置换确定性特征预测结果
#         if (ihd_pred_label[i] == "false"
#                 or vhd_pred_label[i] == "false"
#                 or adr_pred_label[i] == "false"
#                 # or tlr_pred_label[i] == "false"
#                 or cmc_pred_label[i] == "false"
#                 or fc_pred_label[i] == "false"):
#             ihd_pred_label[i] = [1 * ihd_weight, 0]
#             vhd_pred_label[i] = [1 * vhd_weight, 0]
#             adr_pred_label[i] = [1 * adr_weight, 0]
#             cmc_pred_label[i] = [1 * cmc_weight, 0]
#             # tlr_pred_label[i] = [1 * tlr_weight, 0]
#             fc_pred_label[i] = [1 * fc_weight, 0]
#         else:
#             ihd_pred_label[i] = [446 / 1227, 781 / 1227]
#             vhd_pred_label[i] = [621 / 1420, 799 / 1420]
#             adr_pred_label[i] = [462 / 1255, 793 / 1255]
#             cmc_pred_label[i] = [91 / 257, 166 / 257]
#             # tlr_pred_label[i] = [72 / 337, 265 / 337]
#             fc_pred_label[i] = [715 / 1516, 799 / 1516]
#         # 置换非确定性特征预测结果
#         if sc_pred_label[i] == "false":
#             sc_pred_label[i] = [331 / 480 * sc_weight, 149 / 480 * sc_weight_true]
#         else:
#             sc_pred_label[i] = [469 / 1120, 651 / 1120]
#         if sr_pred_label[i] == "false":
#             sr_pred_label[i] = [101 / 137 * sr_weight, 36 / 137 * sr_weight_true]
#         else:
#             sr_pred_label[i] = [59 / 183, 124 / 183]
#
#     joint_pred_label = []
#     for i in range(0, len(reviews)):
#         temp = []
#         pro = (ihd_pred_label[i][0]
#                + vhd_pred_label[i][0]
#                + adr_pred_label[i][0]
#                + cmc_pred_label[i][0]
#                # + tlr_pred_label[i][0]
#                + fc_pred_label[i][0]
#                + sr_pred_label[i][0]
#                + sc_pred_label[i][0])
#         temp.append(pro)
#         pro = (ihd_pred_label[i][1]
#                + vhd_pred_label[i][1]
#                + adr_pred_label[i][1]
#                + cmc_pred_label[i][1]
#                # + tlr_pred_label[i][1]
#                + fc_pred_label[i][1]
#                + sr_pred_label[i][1]
#                + sc_pred_label[i][1])
#         temp.append(pro)
#         joint_pred_label.append(temp)
#     for i, pro in enumerate(joint_pred_label):
#         if pro[0] > pro[1]:
#             joint_pred_label[i] = 1
#         else:
#             joint_pred_label[i] = 0
#
#     return joint_pred_label


def test_method(reviews:list):

    """
    测试方法
    :param reviews:
    :return:
    """

    ihd_pred_label = img_hash_duplication(reviews)
    vhd_pred_label = vid_hash_duplication(reviews)
    cmc_pred_label = color_mode_cls(reviews)
    fc_pred_label = format_cls(reviews)
    adr_pred_label = all_diff_resolution(reviews)
    tlr_pred_label = too_lg_resolution(reviews)

    pred_label = ihd_pred_label.copy()

    # for index, label in enumerate(pred_label):
    #     if label == "false":
    #         pred_label[index] = 1
    #     else:
    #         pred_label[index] = 0
    #
    # return pred_label

    # for index, label in enumerate(pred_label):
    #     if label != vhd_pred_label[index]:
    #         pred_label[index] = "false"
    #
    # for index, label in enumerate(pred_label):
    #     if label != cmc_pred_label[index]:
    #         pred_label[index] = "false"
    #
    # for index, label in enumerate(pred_label):
    #     if label != fc_pred_label[index]:
    #         pred_label[index] = "false"

    for index, label in enumerate(pred_label):
        if label != adr_pred_label[index]:
            pred_label[index] = "false"

    for index, label in enumerate(pred_label):
        if label != tlr_pred_label[index]:
            pred_label[index] = "false"

    ihd_pro = []
    vhd_pro = []
    cmc_pro = []
    fc_pro = []
    sc_pro = []
    adr_pro = []
    sr_pro = []
    tlr_pro = []

    ihd_weight_false = 1 + 354 / 2628
    vhd_weight_false = 1 + 254 / 2628
    adr_weight_false = 1 + 338 / 2628
    cmc_weight_false = 1 + 179 / 2628
    tlr_weight_false = 1 + 584 / 2628
    fc_weight_false = 1 + 83 / 2628
    sr_weight_false = 1 + 505 / 2628
    sc_weight_false = 1 + 331 / 2628

    sr_weight_true = 1 + 180 / 366
    sc_weight_true = 1 + 149 / 366

    ihd_weight = 1
    vhd_weight = 1
    adr_weight = 1
    cmc_weight = 1
    tlr_weight = 1
    fc_weight = 1
    sr_weight = 1
    sc_weight = 1

    sr_weight_t = 1
    sc_weight_t = 1

    for i, review in enumerate(reviews):

        if pred_label[i] == "false":
            reviews[i] = "finished"
            ihd_pro.append([1 * ihd_weight_false, 0])
            vhd_pro.append([1 * vhd_weight_false, 0])
            adr_pro.append([1 * adr_weight_false, 0])
            cmc_pro.append([1 * cmc_weight_false, 0])
            tlr_pro.append([1 * tlr_weight_false, 0])
            fc_pro.append([1 * fc_weight_false, 0])
        else:
            ihd_pro.append([446 / 1227, 781 / 1227])
            vhd_pro.append([621 / 1420, 799 / 1420])
            adr_pro.append([462 / 1255, 793 / 1255])
            cmc_pro.append([91 / 257, 166 / 257])
            tlr_pro.append([72 / 337, 265 / 337])
            fc_pro.append([715 / 1516, 799 / 1516])

    sc_pred_label = size_cls(reviews)
    sr_pred_label = same_resolution(reviews)

    for i in range(0, len(pred_label)):
        if sc_pred_label[i] == "false":
            sc_pro.append([331 / 480 * sc_weight_false, 149 / 480 * sc_weight_true])
        else:
            sc_pro.append([469 / 1120, 651 / 1120])
        if sr_pred_label[i] == "false":
            sr_pro.append([101 / 137 * sr_weight_false, 36 / 137 * sr_weight_true])
        else:
            sr_pro.append([59 / 183, 124 / 183])

    pred_label = []

    for i, _ in enumerate(reviews):
        temp = []
        # pro = (ihd_pro[i][0] + vhd_pro[i][0] + fc_pro[i][0] + cmc_pro[i][0] + sc_pro[i][0])
        pro = (adr_pro[i][0] + tlr_pro[i][0] + sr_pro[i][0])
        temp.append(pro)
        # pro = (ihd_pro[i][1] + vhd_pro[i][1] + fc_pro[i][1] + cmc_pro[i][1] + sc_pro[i][1])
        pro = (adr_pro[i][1] + tlr_pro[i][1] + sr_pro[i][1])
        temp.append(pro)
        pred_label.append(temp)

    for i, pro in enumerate(pred_label):
        if pro[0] > pro[1]:
            pred_label[i] = 1
        else:
            pred_label[i] = 0

    return pred_label