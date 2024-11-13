
import xml.etree.ElementTree as ET
import pickle
import os, shutil, cv2
from os import listdir, getcwd
from os.path import join
import glob
import yaml
#
# classes = [ "Particle",  "Missing", "Residue",  "Bridge",  "Tiny", "Bubble",
#             "Crystal", "Discolor","Scratch","COP", "Grain", "Peeling", "W Puddle"   ]

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



def convert_doc(yaml_file, path, savepath):
    classes = get_classnames(yaml_file)
    filenames = os.listdir(path)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for image_name in filenames:
        in_file = open(os.path.join(path, image_name), 'r', encoding='utf-8')
        xml_text = in_file.read()
        root = ET.fromstring(xml_text)
        in_file.close()

        labeled = root.find('labeled').text
        if labeled == "false":
            continue
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        out_file = open(os.path.join(savepath, image_name[:-4] + '.txt'), 'w', encoding='utf-8')
        for obj in root.iter('item'):
            cls = obj.find('name').text
            if cls not in classes:
                print(cls)
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.close()







if __name__ == "__main__":

    data_yaml_file = 'E:\\v10_v2\\yolov10-pcb-detect\\datasets\\SN01.yaml'
    xml_path = 'E:\\v10_v2\\yolov10-pcb-detect\\datasets\\datatools\\data_examples\\xmls'
    # xml_path = 'E:\\v10\\SN01_dst_2\\xmls'
    txt_path = xml_path + '_yolotxts'
    tag = 1
    if tag:
        convert_annotation(data_yaml_file, xml_path, txt_path)
    else:
        convert_doc(data_yaml_file, xml_path, txt_path)


