import distutils.command.clean
import shutil
import cv2, os, glob, yaml
from ultralytics import YOLOv10 # Note the "v10" in the end
def get_classnames(yaml_file):
    name_list = []
    file = open(yaml_file, 'r', encoding="utf-8")
    # file_data = file.read()
    file_data = yaml.load(file, yaml.FullLoader)
    file.close()
    names_dict = file_data["names"]
    for k,v in names_dict.items():
        # print(v)
        name_list.append(v)
    return name_list



def check_per_class(yaml_file, yolo_outputs, all_class_dirs):
    class_names = get_classnames(yaml_file)
    # img_files = os.listdir(imgs_path)

    yolofiles = os.listdir(yolo_outputs)
    labels_path = os.path.join(yolo_outputs, "labels")
    yolofiles_list = []
    det_cnt, not_det_cnt = 0, 0
    det_files =  [ txt[:-4] for txt in os.listdir(labels_path)]
    for file in yolofiles:
        file_path = os.path.join(yolo_outputs,file)
        if os.path.isfile(file_path):
            yolofiles_list.append(file[-4])
            if file[:-4] in det_files:
                txt_name = file.replace('.jpg', '.txt')
                txt_file = open(os.path.join(labels_path, txt_name), 'r', encoding='utf-8')
                lines = txt_file.readlines()
                txt_file.close()
                pred_cls = lines[0].strip().split()[0]
                pred_classname = class_names[int(pred_cls)]
                img_fold = os.path.join(all_class_dirs, pred_classname)
                if not os.path.exists(img_fold):
                    os.makedirs(img_fold)
                shutil.copy(file_path, os.path.join(img_fold, file))

                det_cnt += 1
            else:
                not_detected_fold = os.path.join(all_class_dirs, "not_detected")
                if not os.path.exists(not_detected_fold):
                    os.makedirs(not_detected_fold)
                shutil.copy(file_path, os.path.join(not_detected_fold,file))
                not_det_cnt += 1

    print("total-num", len(yolofiles_list))
    print("detected-num", det_cnt)
    print("not-detected-num", not_det_cnt)

def yolo_relabel(root,dst,yololab):
    imgs_data = os.path.join(root, "images", "val")
    file_list = [os.path.join(yololab, lab) for lab in os.listdir(yololab)]
    for txt_file in file_list:
        name = os.path.basename(txt_file)[:-4]
        img_path = os.path.join(imgs_data, name + '.jpg')
        new_imgfold = os.path.split(img_path)[0].replace(root,dst)
        new_labfold = new_imgfold.replace("images", "labels")
        if not os.path.exists(new_imgfold):
            os.makedirs(new_imgfold)
        if not os.path.exists(new_labfold):
            os.makedirs(new_labfold)
        shutil.copy(img_path, os.path.join(new_imgfold, name + '.jpg'))

        f = open(txt_file, 'r', encoding="utf-8")
        new_txtfile = os.path.join(new_labfold, name + '.txt')
        new_f = open(new_txtfile, 'w', encoding="utf-8")
        lines = f.readlines()
        f.close()
        for line in lines:
            words = line.strip().split()[:-1]
            print(" ".join(words))
            new_f.write(" ".join(words) + '\n')
        new_f.close()

        # shutil.copy(txt_file, new_txtfile)



if __name__ == '__main__':


    weight_path = 'yolov10n-SN0102-640-1118-1/exp/weights/best.pt'
    imgs_dir = "datasets/SN0102/images/val"
    yolo = YOLOv10(weight_path, task="detect")
    result = yolo(source=imgs_dir,
                  save=True,
                  conf=0.25,
                  iou=0.45,
                  save_conf=True,
                  save_txt=True,
                  name=imgs_dir + '_v10_output')

