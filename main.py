from unsupervised_algorithm import *


if __name__ == '__main__':

    reviews_path = r"D:\dataset\reviews"
    reviews, label = read_data(reviews_path)

    mean_acc = 0
    mean_precision = 0
    mean_recall = 0
    mean_f1 = 0
    mean_f05 = 0
    mean_f2 = 0
    mean_auc_area = 0

    for _ in range(0, 1):
        (accu, precision, recall, f1, f05, f2, auc_area) = k_fold_cross_validation(reviews
                                                        , label
                                                        , k=5
                                                        , shuffle=True
                                                        # , random_state=2
                                                        , method="top8"
                                                        )
        mean_acc += accu
        mean_precision += precision
        mean_recall += recall
        mean_f1 += f1
        mean_f05 += f05
        mean_f2 += f2
        mean_auc_area += auc_area

    print("mean acc:{}".format(mean_acc / 1))
    print("mean precision:{}".format(mean_precision / 1))
    print("mean recall:{}".format(mean_recall / 1))
    print("mean f1:{}".format(mean_f1 / 1))
    # print("mean f05:{}".format(mean_f05 / 1))
    # print("mean f2:{}".format(mean_f2 / 1))
    print("mean auc_area:{}".format(mean_auc_area / 1))
