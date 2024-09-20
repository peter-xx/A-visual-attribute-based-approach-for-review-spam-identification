import glob
import hashlib
import os.path
import shutil
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC
import matplotlib.pyplot as plt


def sim_cls(reviews:list, is_huawei:bool=True):

    """根据评论中图片间的相似度判断虚假评论"""

    pred_label = reviews.copy()

    if is_huawei:
        threshold = 0.14
        # threshold = 0.6
    else:
        threshold = 0.23
        # threshold = 0.6

    for index, review in enumerate(reviews):

        imgs = get_images(review)

        similarity = get_hist(imgs[0], imgs[1], cv2.HISTCMP_CORREL)
        # similarity = get_meanhash(imgs[0], imgs[1], 64)

        if similarity < threshold:
            pred_label[index] = "false"

    return pred_label


def img_wtm_cls(reviews:list, output_path:str):

    """通过图片中的水印判断虚假评论"""

    pred_label = reviews.copy()

    weight_path = r"E:/DataSets/review_dataset/watermark_dataset/exp/fine/new_aug/vit-b-16/model-15.pth"

    pred_label = wtm_cls(reviews=reviews, pred_label=pred_label, output_path=output_path, weight_path=weight_path, crop_region="middle")

    pred_label = wtm_cls(reviews=reviews, pred_label=pred_label, output_path=output_path, weight_path=weight_path, crop_region="lower_right")

    pred_label = wtm_cls(reviews=reviews, pred_label=pred_label, output_path=output_path, weight_path=weight_path, crop_region="upper_right")

    return pred_label


def vid_wtm_cls(reviews:list, output_path:str):

    """通过视频中的水印判断虚假评论"""

    pred_label = reviews.copy()
    weight_path = r"E:/DataSets/review_dataset/watermark_dataset/exp/fine/new_aug/vit-b-16/model-15.pth"

    for index, review in enumerate(reviews):

        video = get_video(review)

        if len(video) > 0:

            pred_classes = crop_pred(video[0], output_path, weight_path)

            for pred_class in pred_classes:
                if pred_class[0] == 1 or pred_class[1] == 1 or pred_class[2] == 1:
                    pred_label[index] = "false"

    return pred_label


def draw_roc(label:list, pred_prob:list, pos_label, is_draw):

    FPR, recall, thresholds = roc_curve(y_true=label, y_score=pred_prob, pos_label=pos_label)
    area = AUC(y_true=label, y_score=pred_prob) * 100

    if is_draw:
        plt.figure()
        plt.plot(FPR, recall, color='red',
                 label='ROC curve (area = %0.2f)' % area)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([-0.05, 1.05])  # 不是在0，1是因为怕挤着不太好看
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('Recall')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return area


def model_metrics(label, pred_label):

    precision = precision_score(label, pred_label, pos_label=1)

    recall = recall_score(label, pred_label, pos_label=1)

    f1 = f1_score(label, pred_label, pos_label=1)

    return precision, recall, f1


def average_hash(image_path, hash_size:int=8):

    # 打开图像并转换为灰度
    image = Image.open(image_path).convert('L')

    # 缩放图像到指定的哈希尺寸
    image = image.resize((hash_size, hash_size), Image.ANTIALIAS)

    # 计算像素平均值
    pixels = list(image.getdata())
    average_pixel = sum(pixels) / len(pixels)

    # 生成哈希
    hash_value = ''.join(['1' if pixel > average_pixel else '0' for pixel in pixels])

    return hash_value

def hamming_distance(hash1, hash2):

    # 计算汉明距离
    return sum([1 for a, b in zip(hash1, hash2) if a != b])


def get_meanhash(img1:str, img2:str, hash_size:int):

    hash1 = average_hash(img1, hash_size)
    hash2 = average_hash(img2, hash_size)
    distance = hamming_distance(hash1, hash2)
    similarity = 1 - (distance / len(hash1))

    return similarity


def get_hist(image1, image2, method=cv2.HISTCMP_BHATTACHARYYA):

    # 读取图像
    img1 = cv2.imdecode(np.fromfile(image1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.fromfile(image2, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    img1 = cv2.resize(img1, (304, 405), interpolation=cv2.INTER_NEAREST)  # resize images
    img2 = cv2.resize(img2, (304, 405), interpolation=cv2.INTER_NEAREST)  # resize images

    # 计算直方图
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    # 归一化直方图
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # 使用巴氏距离进行直方图比较
    similarity = cv2.compareHist(hist1, hist2, method)

    return similarity


def change_to_review(images:list, source_path, target_path):

    """将图片变成评论的形式"""

    for i in range(len(images)):

        image = images[i]
        image_path = source_path + "/" + image
        target_file_path = target_path + "/" + str(i)

        if not os.path.exists(target_file_path):
            os.mkdir(target_file_path)

        target_image_path = target_file_path + "/" + image

        if not os.path.exists(target_image_path):
            shutil.move(image_path, target_file_path)
        else:
            file_name, file_extension = os.path.splitext(target_image_path)
            new_path = f"{file_name}(1){file_extension}"
            shutil.move(image_path, new_path)

    print("changed!")


def torgb(source_file, target_file):

    """
    将文件夹内的图片转为RGB模式
    :param source_file: 需要转换的文件夹
    :param target_file: 转换后存储的文件夹
    """

    assert os.path.exists(source_file), "source file: {} does not exist.".format(source_file)

    for image in os.listdir(source_file):

        image_path = os.path.join(source_file, image)

        img = Image.open(image_path)

        if (img.mode != 'RGB'):
            img = img.convert("RGB")
            print(image)
            img.save(target_file + '\\' + image)

    print("converted!")


def custom_sort(element):

    """按照最后一个//后的内容排序"""

    return int(element.split("\\")[-1])


def custom_int(element):
    return int(element)


def get_image_size(reviews:list):

    """"获取每条评论的图片大小，存放在两个列表中"""

    max_img_sizes = []
    min_img_sizes = []

    for index, review in enumerate(reviews):

        if review != "finished":
            imgs = get_images(review)

            # 返回的是字节
            first_img_size = os.path.getsize(imgs[0]) / 1024

            second_img_size = os.path.getsize(imgs[1]) / 1024

            if first_img_size > second_img_size:
                max_img_sizes.append(first_img_size)
                min_img_sizes.append(second_img_size)
            else:
                max_img_sizes.append(second_img_size)
                min_img_sizes.append(first_img_size)
        else:
            max_img_sizes.append("finished")
            min_img_sizes.append("finished")

    return max_img_sizes, min_img_sizes


def read_data(root:str):

    """读取数据集，返回数据和标签"""

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    data_class = ["false", "true"]

    reviews = []
    reviews_label = []

    for cla in data_class:

        cla_path =os.path.join(root, cla)

        review_list = os.listdir(cla_path)
        review_list.sort(key=custom_int)

        for review in review_list:
            reviews.append(os.path.join(cla_path, review))
            if cla == "false":
                reviews_label.append(1)
            else:
                reviews_label.append(0)

    return reviews, reviews_label


def get_images(folder_path):

    image_list = glob.glob(os.path.join(folder_path, '*.jpg')) + \
                  glob.glob(os.path.join(folder_path, '*.png')) + \
                  glob.glob(os.path.join(folder_path, '*.jpeg')) + \
                  glob.glob(os.path.join(folder_path, '*.webp'))

    return image_list


def to_hash(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        hash_value = hashlib.md5(data).hexdigest()

    return hash_value


def accuracy(label, pred_label):

    count = len(label)

    accu_num = 0

    for index, cla in enumerate(label):

        if cla == 1 and pred_label[index] == 1:
            accu_num += 1
        elif cla == 0 and pred_label[index] == 0:
            accu_num += 1
        # else:
        #     print(index)

    accu = accu_num / count

    return accu


def get_video(folder_path):

    video = glob.glob(os.path.join(folder_path, '*.mp4')) + \
                  glob.glob(os.path.join(folder_path, '*.avi')) + \
                  glob.glob(os.path.join(folder_path, '*.mov'))
    return video


def extract_elements(input_list, indices):
    return [input_list[i] for i in indices]


def result_same(result1, result2):

    """判断两个预测结果是否相同"""

    has_error = False

    for index, label in enumerate(result1):

        if label == "false":

            if result2[index] != "false":
                has_error = True
                return has_error

    for index, label in enumerate(result2):

        if label == "false":

            if result1[index] != "false":
                has_error = True
                return has_error

    return has_error


def crop_and_save(image, output_path_base, crop_region:str, name:str):

    # 获取图片的宽度和高度
    height, width, _ = image.shape

    # 定义裁剪的区域
    middle_box = (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
    upper_right_box = (2 * width // 3, 0, width, height // 3)
    lower_right_box = (2 * width // 3, 2 * height // 3, width, height)
    upper_left_box = (0, 0, width // 3, height // 3)
    lower_left_box = (0, 2 * height // 3, width // 3, height)

    # 根据定义的区域进行裁剪
    if crop_region == "middle":
        middle_image = image[middle_box[1]:middle_box[3], middle_box[0]:middle_box[2]]
        cv2.imwrite(os.path.join(output_path_base, name + '_middle.jpg'), middle_image)
    elif crop_region == "upper_right":
        upper_right_image = image[upper_right_box[1]:upper_right_box[3], upper_right_box[0]:upper_right_box[2]]
        cv2.imwrite(os.path.join(output_path_base, name + '_upper_right.jpg'), upper_right_image)
    elif crop_region == "lower_right":
        lower_right_image = image[lower_right_box[1]:lower_right_box[3], lower_right_box[0]:lower_right_box[2]]
        cv2.imwrite(os.path.join(output_path_base, name + '_lower_right.jpg'), lower_right_image)
    elif crop_region == "upper_left":
        upper_left_image = image[upper_left_box[1]:upper_left_box[3], upper_left_box[0]:upper_left_box[2]]
        cv2.imwrite(os.path.join(output_path_base, name + '_upper_left.jpg'), upper_left_image)
    else:
        lower_left_image = image[lower_left_box[1]:lower_left_box[3], lower_left_box[0]:lower_left_box[2]]
        cv2.imwrite(os.path.join(output_path_base, name + '_lower_left.jpg'), lower_left_image)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image


def model_predict_watermark(data:list, weight_path:str, batch_size:int=8):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = CustomDataset(data, data_transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True)
    model = models.vit_b_16(weights="IMAGENET1K_V1")
    model.heads.head = nn.Linear(model.heads.head.in_features, out_features=2)
    model.to(device=device)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    pred_classes = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device=device)
            pred = model(batch)
            pred_classes.append(torch.max(pred, dim=1)[1])

    return pred_classes


def wtm_cls(reviews:list, pred_label:list, output_path:str, weight_path:str, crop_region):

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    for i, review in enumerate(reviews):

        if pred_label[i] != "false":
            imgs = get_images(review)

            for j, img in enumerate(imgs):
                img = np.array(cv2.imread(img))
                crop_and_save(img, output_path, crop_region, str(i) + str(j))

    dataset = [os.path.join(output_path, image) for image in os.listdir(output_path)]

    pred_classes = model_predict_watermark(data=dataset, batch_size=2, weight_path=weight_path)

    for index, pred_class in enumerate(pred_classes):

        if pred_class[0] == 1 or pred_class[1] == 1:

            image = dataset[index*2]
            image_name = image.split("\\")[-1]
            false_index = int(image_name[0])
            pred_label[false_index] = "false"

    return pred_label


def extract_frame(video_path, output_folder):

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 提取中间帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()

    # 释放视频对象
    cap.release()

    return frame


def crop_pred(video, output_path, weight_path):

    img = extract_frame(video_path=video, output_folder=output_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    crop_and_save(img, output_path, "middle", "0")
    crop_and_save(img, output_path, "upper_right", "0")
    crop_and_save(img, output_path, "lower_right", "0")
    dataset = [os.path.join(output_path, image) for image in os.listdir(output_path)]
    pred_classes = model_predict_watermark(data=dataset, batch_size=3, weight_path=weight_path)

    return pred_classes