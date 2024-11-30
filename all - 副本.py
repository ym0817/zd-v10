# coding=utf-8
from multiprocessing.connection import families

import torch
from humanfriendly.terminal import find_terminal_size
from scipy.stats import alpha
from ultralytics import YOLOv10
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, shutil, cv2
import time, csv
import yaml
import random
import argparse



def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    final_x = 0.0 if x < 0.0 else x
    final_x = 1.0 if final_x > 1.0 else x

    final_y = 0.0 if y < 0.0 else y
    final_y = 1.0 if final_y > 1.0 else y

    final_w = 0.0 if w < 0.0 else w
    final_w = 1.0 if final_w > 1.0 else w

    final_h = 0.0 if h < 0.0 else h
    final_h = 1.0 if final_h > 1.0 else h
    return (final_x, final_y, final_w, final_h)



def convert_annotation(classes, path, savepath):
    filenames = os.listdir(path)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for image_name in filenames:
        # print(image_name)
        in_file = open(os.path.join(path, image_name), 'r', encoding='utf-8')
        xml_text = in_file.read()
        root = ET.fromstring(xml_text)
        in_file.close()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        out_file = open(os.path.join(savepath, image_name[:-4] + '.txt'), 'w', encoding='utf-8')
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                print('Not exist in Classes  ' ,image_name, cls)
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.close()

def check_image_txt_pair( file_path, txt_path ):
    all_images,  all_labels = [], []
    for label in os.listdir(txt_path):
        label_path = os.path.join(txt_path, label)
        in_file = open(label_path, 'r', encoding='utf-8')
        lines = in_file.readlines()
        in_file.close()
        if len(lines):
            image = label.replace('.txt', '.jpg')
            image_path = os.path.join(file_path, image)
            im = cv2.imread(image_path)
            if im is not None:
                all_images.append(image)
                all_labels.append(label)
    return all_images,  all_labels


def split_data(file_path, txt_path, new_file_path, types, ratios):
    all_images, all_labels = check_image_txt_pair(file_path,txt_path)
    assert len(all_images) == len(all_labels)
    total = len(all_images)
    print('total image-txt-pair num : ', total)

    data = list(zip(all_images, all_labels))
    random.shuffle(data)
    each_class_image, each_class_label = zip(*data)

    index_list = [0]
    cnt = 0
    for r in range(len(ratios)-1):
        cnt += int(total * ratios[r])
        index_list.append(cnt)
    index_list.append(total)
    print(index_list)
    for i in range(len(types)):
        start_index, end_index = index_list[i], index_list[i+1]
        images = each_class_image[start_index:end_index]
        labels = each_class_label[start_index:end_index]
        new_imgfoldpath = os.path.join(new_file_path, 'images',types[i])
        if not os.path.exists(new_imgfoldpath):
            os.makedirs(new_imgfoldpath)
        new_txtfoldpath = os.path.join(new_file_path, 'labels', types[i])
        if not os.path.exists(new_txtfoldpath):
            os.makedirs(new_txtfoldpath)
        c = 0
        for image,label in zip(images, labels):
            old_imgpath = os.path.join(file_path, image)
            old_txtpath = os.path.join(txt_path, label)
            new_imgpath = os.path.join(new_imgfoldpath, image)
            new_txtpath = os.path.join(new_txtfoldpath, label)
            shutil.copy(old_imgpath, new_imgpath)
            shutil.copy(old_txtpath, new_txtpath)
            c += 1
        print(types[i] + ' image-txt-pair num : ', c)



def train(model, data_root, batch_size, img_size, max_epoch, save_weight):
    results = model.train(
        data=data_root,
        epochs=max_epoch,
        imgsz=img_size,  # imgsz应该尽量贴近训练集图片的大小，但是要是32的倍数
        plots=True,
        batch=batch_size,
        amp=False,
        # fraction=0.1, # 设置fraction参数用于只训练数据集的一部分，设置0.1表示只训练10%的数据集
        # project= project_name,
        patience=30,
        # degrees=180,
        auto_augment="autoaugment",
        cache=True,  # 缓存数据集，加快训练速度
        hsv_h=0.02,  # 调整图像的色调，对提高模型对不同颜色的物体的识别能力有帮助
        translate=0.2,  # 平移图像，帮助模型识别边角的物体
        flipud=0.5,  # 以指定概率上下翻转图像
        # bgr=0.5, # 以指定概率随机改变图像的颜色通道顺序，提高模型对不同颜色的物体的识别能力
        close_mosaic=20,  # 最后稳定训练
        scale=0.3,
        device=0,
        workers=0,
        save_dir = save_weight,
        # name=save_weight,
        resume=False,  # 断点训练，默认Flase

    )
    model.export(format='onnx', opset=11)

def evaluation(model, data_root, batch_size, img_size):
    # 加载模型，split='test'利用测试集进行测试
    model.val(data=data_root,
              # split='test',
              imgsz=img_size,
              batch=batch_size,
              device=0,
              workers=0,
              save=True,
              task="test")


def check_charset(file_path):
    import chardet
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset

def over_message():
    width, height = 640, 480
    blank_image = np.zeros((height, width, 3), np.uint8)
    text = "over"
    position = (width // 2 - 20, height // 2 + 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    background_color = (0, 0, 0)  # 黑色
    cv2.putText(blank_image, text, position, font, font_scale, font_color, 2)
    cv2.imwrite("Over.jpg", blank_image)

def DisplaySampleInfo(data_path, classname_list, resilt_dir):
    types = ['train','val']
    classname_list = [c+1 for c in range(len(classname_list))]
    train_cnt_list = [0 for i in range(len(classname_list))]
    val_cnt_list = [0 for i in range(len(classname_list))]
    train_sample_cnt, val_sample_cnt = 0, 0
    for type in types:
        labdir_path = os.path.join(data_path, "labels", type)
        for lab in os.listdir(labdir_path):
            lab_path = os.path.join(labdir_path, lab)
            in_file = open(lab_path, 'r', encoding='utf-8')
            lines = in_file.readlines()
            in_file.close()
            if type == 'train':
                train_sample_cnt += 1
            else:
                val_sample_cnt += 1
            for line in lines:
                cls_id = int(line.strip().split()[0])
                if type == 'train':
                    train_cnt_list[cls_id] += 1
                else:
                    val_cnt_list[cls_id] += 1
        # print(classname_list)
        # print(cnt_list)
    fig = plt.figure()
    plt.rc('font', family = 'simhei',size = 15)
    plt.rc('axes', unicode_minus = False)
    classname_list_x = np.arange(len(classname_list)) * 0.8
    t_name = 'train-imgs (' + str(train_sample_cnt) + ')'
    v_name = 'val-imgs (' + str(val_sample_cnt) + ')'
    plt.bar(classname_list_x, train_cnt_list, width=0.2, align='edge', alpha = 0.8, label=t_name, color='skyblue')
    classname_list_xv = [x0 + 0.2 for x0 in classname_list_x]
    plt.bar(classname_list_xv, val_cnt_list, width=0.2, align='edge', alpha=0.8, label=v_name, color='blue')
    plt.title('Train-Val Sample Count')
    # plt.legend(['recall'], loc='upper right')
    plt.legend()
    plt.ylim(0.8)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks([x0 + 0.2 for x0 in classname_list_x], classname_list)
    # plt.show()
    resimg_path = os.path.join(resilt_dir, 'Train-Val Sample Count.png')
    plt.savefig(resimg_path)


        # ax1.set_ylabel('count')
        # plt.legend(loc='upper left')
        # # plt.savefig("/home/scsc/dataset/log1.png")
        # plt.show()


def DisplayProcessInfo(model_path,resilt_dir):
    csv_file = os.path.join(model_path, "results.csv")
    df = pd.read_csv(csv_file)
    labels = list(df.columns.values)
    print(labels)
    lines = df.values.tolist()
    data = np.array(df.loc[:,:])
    box_om_loss = df['           train/box_om'].tolist()
    cls_om_loss = df['           train/cls_om'].tolist()
    dfl_om_loss = df['           train/dfl_om'].tolist()
    box_oo_loss = df['           train/box_oo'].tolist()
    cls_oo_loss = df['           train/cls_oo'].tolist()
    dfl_oo_loss = df['           train/dfl_oo'].tolist()

    epoch_list = df['                  epoch'].tolist()
    p_list = df['   metrics/precision(B)'].tolist()
    r_list = df['      metrics/recall(B)'].tolist()
    mAP50_list = df['       metrics/mAP50(B)'].tolist()
    # epoch_listoo = np.array(df.iloc[:,:1]).tolist()
    # print(epoch_list)
    fig = plt.figure(1)
    plt.plot(epoch_list, p_list,label='precision', color='gold')  # 注意bottom参数用于堆叠条形图
    plt.legend(['precision'], loc= 'upper right')
    plt.title('Precision-Iterations Curve')
    plt.xlabel('iterations')
    plt.ylabel('precision')
    # plt.show()
    precision_img_path = os.path.join(resilt_dir, 'Precision-Iterations Curve.png')
    plt.savefig(precision_img_path)

    fig = plt.figure(2)
    plt.plot(epoch_list, r_list, label='recall', color='blue')  # 注意bottom参数用于堆叠条形图
    plt.legend(['recall'], loc='upper right')
    plt.title('Recall-Iterations Curve')
    plt.xlabel('iterations')
    plt.ylabel('recall')
    # plt.show()
    recall_img_path = os.path.join(resilt_dir, 'Recall-Iterations Curve')
    plt.savefig(recall_img_path)

    fig = plt.figure(3)
    plt.plot(epoch_list, mAP50_list, label='mAP50', color='green')  # 注意bottom参数用于堆叠条形图
    plt.legend(['mAP50'], loc='upper right')
    plt.title('mAP50-Iterations Curve')
    plt.xlabel('iterations')
    plt.ylabel('mAP50')
    # plt.show()
    mAP50_img_path = os.path.join(resilt_dir, 'mAP50-Iterations Curve')
    plt.savefig(mAP50_img_path)

    fig,ax = plt.subplots(2,3,figsize = (9,6))
    ax[0][0].plot(epoch_list, box_om_loss, label='box_om_loss', color='green')
    ax[0][0].set_xlabel('iterations',fontsize=12)
    ax[0][0].set_ylabel('box_om_loss',fontsize=12)

    ax[0][1].plot(epoch_list, cls_om_loss, label='cls_om_loss', color='gray')
    ax[0][1].set_xlabel('iterations')
    ax[0][1].set_ylabel('cls_om_loss')

    ax[0][2].plot(epoch_list, dfl_om_loss, label='dfl_om_loss', color='red')
    ax[0][2].set_xlabel('iterations')
    ax[0][2].set_ylabel('dfl_om_loss')

    ax[1][0].plot(epoch_list, box_oo_loss, label='box_oo_loss', color='blue')
    ax[1][0].set_xlabel('iterations')
    ax[1][0].set_ylabel('box_oo_loss')

    ax[1][1].plot(epoch_list, cls_oo_loss, label='cls_oo_loss', color='gold')
    ax[1][1].set_xlabel('iterations')
    ax[1][1].set_ylabel('cls_oo_loss')

    ax[1][2].plot(epoch_list, dfl_oo_loss, label='dfl_oo_loss', color='pink')
    ax[1][2].set_xlabel('iterations')
    ax[1][2].set_ylabel('dfl_oo_loss')
    # plt.show()
    loss_img_path = os.path.join(resilt_dir, 'mAP50-Iterations Curve')
    plt.savefig(mAP50_img_path)







def DisplayResults(model_path, data_path, classname_list, resilt_dir):
    DisplayProcessInfo(model_path, resilt_dir)
    DisplaySampleInfo(data_path, classname_list, resilt_dir)


def main(yaml_file):
    # file = open(yaml_file, 'r', encoding="utf-8")
    file = open(yaml_file, 'r', encoding="UTF-8")

    file_data = yaml.load(file, Loader = yaml.FullLoader)
    # file_data = yaml.safe_load(file)
    file.close()
    abs_path = file_data["path"]
    samples_path = file_data["samples_path"]
    xlms_path = file_data["xlms_path"]
    save_modelpath = file_data["save_modelpath"]
    model_backup = file_data["backup"]
    init_weight_path = file_data["init_weight_path"]
    data_name = file_data["data_name"]
    dataset_types = file_data["dataset_types"]
    dataset_ratio = file_data["dataset_ratio"]
    batch_size = file_data["batch_size"]
    img_size = file_data["img_size"]
    max_epoch = file_data["max_epoch"]

    names_dict = file_data["names"]
    name_list = []
    for k, v in names_dict.items():
        print(v)
        name_list.append(v)
    imgs_data = os.path.join(abs_path, samples_path)
    xmls_data = os.path.join(abs_path, xlms_path)
    txts_data = os.path.join(os.path.split(xmls_data)[0], "temp_txts")
    train_path = os.path.join(os.path.split(xmls_data)[0], data_name)
    if os.path.exists(txts_data):
        shutil.rmtree(txts_data)
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    convert_annotation(name_list, xmls_data, txts_data)
    split_data(imgs_data, txts_data, train_path, dataset_types, dataset_ratio)
    # if os.path.exists(txts_data):
    #     shutil.rmtree(txts_data)
    # pretrained_weight = os.path.join(abs_path, init_weight_path)
    # model = YOLOv10(pretrained_weight)
    # if os.path.exists(save_modelpath):
    #     shutil.rmtree(save_modelpath)
    # backup_dir = os.path.join(abs_path, model_backup)
    # if os.path.exists(backup_dir):
    #     shutil.rmtree(backup_dir)
    #
    # train(model, yaml_file, batch_size, img_size, max_epoch, save_modelpath)



    # shutil.copytree(save_modelpath, backup_dir)
    DisplayResults(save_modelpath, train_path, name_list, 'E:\\Work\\SVNs\\AIADC\\Trains\\Trains_V1\\Recipe\\1111')
    # over_message()





if __name__ == '__main__':

    yaml_file = "E:\\Work\\SVNs\\AIADC\\Trains\\Trains_V1\\Recipe\\config.yaml"
    # yaml_file = sys.argv[1]

    start_time = time.time()
    main(yaml_file)
    end_time = time.time()
    print(f"执行时间：{(end_time - start_time) / 3600} Hours")





