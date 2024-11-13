import os, cv2
import shutil
import random

random.seed(0)


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


# def split_data(file_path, txt_path, new_file_path, train_rate, val_rate, test_rate):
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


if __name__ == '__main__':
    file_path = "E:\\v10_v2\\yolov10-pcb-detect\\datasets\\datatools\\data_examples\\samples"
    txt_path = "E:\\v10_v2\\yolov10-pcb-detect\\datasets\\datatools\\data_examples\\xmls_yolotxts"
    save_path = "E:\\v10_v2\\yolov10-pcb-detect\\datasets\\datatools\\data_examples\\SN01_temp"

    dataset__types = ['train', 'val', 'test']
    dataset_ratio = [0.8, 0.1, 0.1]
    split_data(file_path, txt_path, save_path, dataset__types, dataset_ratio)
