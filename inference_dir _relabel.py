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



weight_path = 'yolov10n-SN0102-640-1118-1/exp/weights/best.pt'
yaml_file = 'datasets/SN01.yaml'
dataset = "datasets/SN0102"
new_dataset = dataset + "_reabel"
imgs_dir = os.path.join(dataset, "images" ,"val")
file_list = [ os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir)]

class_names = get_classnames(yaml_file)
model = YOLOv10(weight_path)

cnt = 0
for file_path in file_list:
    img_name = os.path.basename(file_path)
    cls_name = img_name.strip().split('_')[0]
    defect_num = int(img_name.strip().split('_')[1])
    frame = cv2.imread(file_path)
    label_path = file_path.replace('images', 'labels').replace('.jpg', '.txt')
    img_fold = os.path.split(file_path)[0].replace(dataset, new_dataset)
    if not os.path.exists(img_fold):
        os.makedirs(img_fold)
    label_fold = os.path.split(label_path)[0].replace(dataset, new_dataset)
    if not os.path.exists(label_fold):
        os.makedirs(label_fold)

    in_file = open(label_path, 'r', encoding='utf-8')
    lines = in_file.readlines()
    in_file.close()

    if len(lines) > 1:
        shutil.copy(file_path, os.path.join(img_fold, img_name))
        shutil.copy(label_path, os.path.join(label_fold, img_name.replace('.jpg', '.txt')))

    else:
        shutil.copy(file_path, os.path.join(img_fold, img_name))
        new_label_path = os.path.join(label_fold, img_name.replace('.jpg', '.txt'))
        new_labelfile = open(new_label_path, 'w', encoding='utf-8')
        real_cls = int(lines[0].strip().split()[0])
        results = model.predict(frame)
        for result in results:
            # 结果中的每个元素对应一张图片的预测
            boxes = result.boxes  # 获取边界框信息
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                predict_cls = int(box.cls[0])
                conf = float(box.conf[0])
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 3)
                # cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #             (0, 180, 255), 2)

                new_labelfile.write(str(predict_cls) + " " + " ".join(lines[0].strip().split()[1:]))
        new_labelfile.close()
        cnt += 1

        #         if predict_cls == real_cls:
        #             shutil.copy(file_path, os.path.join(img_fold, img_name))
        #             shutil.copy(label_path, os.path.join(label_fold, img_name.replace('.jpg', '.txt')))
        #         else:
        #             # new_img_name = img_name.replace(cls_name, class_names[predict_cls] + '_src' + cls_name)
        #             new_img_name = img_name
        #             shutil.copy(file_path, os.path.join(img_fold, new_img_name))
        #             new_label_path = os.path.join(label_fold, new_img_name.replace('.jpg', '.txt'))
        #             new_labelfile = open(new_label_path, 'w', encoding='utf-8')
        #             new_labelfile.write(str(predict_cls) + " " + " ".join(lines[0].strip().split()[1:]))
        #             new_labelfile.close()

    # else:
    #     shutil.copy(file_path, os.path.join(img_fold, img_name))
    #     shutil.copy(label_path, os.path.join(label_fold, img_name.replace('.jpg', '.txt')))

print("src-img total :", len(file_list))
print("src-txt total :", cnt)






            # result_dir = path[0] + '_results'
            # per_cls_dir = os.path.join(result_dir,class_names[cls])



            # if not os.path.exists(per_cls_dir):
            #     os.makedirs(per_cls_dir)
            # cv2.imwrite(os.path.join(per_cls_dir,os.path.basename(file_path)), frame)








# imgs_path = "datasets/SN0102/images/val/*.jpg"
# labels_path =




# path = os.path.split(imgs_path)
# result_dir = path[0] + '_results'
# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)

# file_list = glob.glob(imgs_path)



    # 显示带有检测结果的帧
    # h, w, c = frame.shape
    # frame = cv2.resize(frame, (int(w * 0.5), int(h * 0.5)))
    # cv2.imshow('YOLOv10', frame)
    # cv2.waitKey(0)
    # img_name = os.path.basename(file_path)
    # result_path = os.path.join(result_dir,img_name)
    # cv2.imwrite(result_path,frame)






# imgpaths = ['1.jpg', '2.jpg', '3.jpg']

# Load a model
# model = YOLOv10('runs/train/exp/weights/best.pt') # load an official model
# Predict with the model
# model.predict(0) # predict on your webcam

# results = model.predict(imgpaths[0]) # predict on your webcam
# results[0].show()


# results = model(source=imgpaths, conf=0.25,save=True)


# import glob
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import matplotlib
# matplotlib.use('TkAgg')
#
# images = glob.glob('runs/detect/predict/*.jpg')
#
# images_to_display = images[:2]
#
# fig, axes = plt.subplots(1, 2, figsize=(20, 10))
#
# for i, ax in enumerate(axes.flat):
#     if i < len(images_to_display):
#         img = mpimg.imread(images_to_display[i])
#         ax.imshow(img)
#         ax.axis('off')
#     else:
#         ax.axis('off')
#
# plt.tight_layout()
# plt.show()
