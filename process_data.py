
import xml.etree.ElementTree as ET
import pickle
import os, shutil, cv2
from os import listdir, getcwd
from os.path import join
import glob
import yaml
import random


def get_classnames(yaml_file):
    name_list = []
    file = open(yaml_file, 'r', encoding="utf-8")
    # file_data = file.read()
    file_data = yaml.load(file, yaml.FullLoader)
    file.close()
    names_dict = file_data["names"]
    for k,v in names_dict.items():
        print(v)
        name_list.append(v)
    return name_list

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
    return (x, y, w, h)


def convert_annotation(yaml_file, path, savepath):
    classes = get_classnames(yaml_file)
    filenames = os.listdir(path)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for image_name in filenames:
        # print(image_name)
        if image_name.endswith('.xml'):
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





if __name__ == "__main__":

    abs_path = "E:\\v10\\SN2\\SN02"
    data_yaml_file = os.path.join(abs_path, "SN01.yaml")
    imgs_data =  os.path.join(abs_path, "images")
    xmls_data = os.path.join(abs_path, "labelimg_xmls")
    txts_data = os.path.join(abs_path, "temp_txts")
    save_path = os.path.join(abs_path, "SN02")

    dataset_types = ['train', 'val', 'test']
    dataset_ratio = [0.9, 0.1, 0.0]
    convert_annotation(data_yaml_file, xmls_data, txts_data)

    split_data(imgs_data, txts_data, save_path, dataset_types, dataset_ratio)




