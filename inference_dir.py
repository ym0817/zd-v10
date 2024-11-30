import cv2, os, glob, yaml,shutil
from ultralytics import YOLOv10 # Note the "v10" in the end
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


weight_path = 'yolov10n-SN0102-640-1118-1/exp/weights/best.pt'
yaml_file = 'datasets/SN01.yaml'
imgs_path = "datasets/SN0102/images/val/*.jpg"


class_names = get_classnames(yaml_file)


path = os.path.split(imgs_path)
result_dir = path[0] + '_results'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
model = YOLOv10(weight_path)
file_list = glob.glob(imgs_path)

for file_path in file_list:
    frame = cv2.imread(file_path)
    results = model.predict(frame)
    if len(results)<1:
        if not os.path.exists(os.path.join(result_dir,"not_detected")):
            os.makedirs(os.path.join(result_dir,"not_detected"))
        shutil.copy(file_path,os.path.join(result_dir,"not_detected",os.path.basename(file_path)))
    else:
        for result in results:
            # 结果中的每个元素对应一张图片的预测
            boxes = result.boxes  # 获取边界框信息
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 3)
                cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 180, 255), 2)

                result_dir = path[0] + '_results'
                per_cls_dir = os.path.join(result_dir,class_names[cls])
                # for i in range(9):
                #     if cls == i:
                #         per_result_dir = path[0] + '_results_' + str(i)
                if not os.path.exists(per_cls_dir):
                    os.makedirs(per_cls_dir)
                cv2.imwrite(os.path.join(per_cls_dir,os.path.basename(file_path)), frame)



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
