import os

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# dlib预测器和OpenCV级联分类器的路径
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)


# 获取面部特征点的函数
def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3, 5)
    x, y, w, h = rects[0]
    rect = dlib.rectangle(x, y, x + w, y + h)
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


# 全局变量定义
modelpath = "models/model.pt"  # 模型权重文件路径
image_folder_path = "testimages"  # 测试图像文件夹路径

testsize = 48  # 测试图像尺寸
from net import ThreeConvNet

net = ThreeConvNet(4)  # 定义网络模型，参数4表示模型输出的类别数
net.eval()  # 设置为评估模式
torch.no_grad()  # 关闭梯度计算，节省计算资源和内存

# 载入模型权重
net.load_state_dict(torch.load(modelpath, map_location=lambda storage, loc: storage))

# 定义图像预处理流程
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# 遍历测试图像进行预测
imagepaths = os.listdir(image_folder_path)
for imagepath in imagepaths:
    im = cv2.imread(os.path.join(image_folder_path, imagepath), 1)
    try:
        # 使用OpenCV检测人脸
        rects = cascade.detectMultiScale(im, 1.3, 5)
        if len(rects) == 0:  # 如果未检测到人脸，则抛出异常
            raise ValueError("No faces detected")
        x, y, w, h = rects[0]
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    except ValueError as e:
        print(f"ValueError: {e}")  # 打印错误信息
        continue
    except Exception as e:
        print(f"An exception occurred: {e}")  # 打印其他异常信息
        continue

    # 根据特征点确定感兴趣区域（ROI）
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0

    for i in range(48, 67):  # 只考虑嘴巴周围的特征点
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y

    roiwidth = xmax - xmin
    roiheight = ymax - ymin

    roi = im[ymin:ymax, xmin:xmax, 0:3]

    # 确定最终ROI的大小
    if roiwidth > roiheight:
        dstlen = 1.5 * roiwidth
    else:
        dstlen = 1.5 * roiheight

    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight

    newx = xmin
    newy = ymin

    imagerows, imagecols, channel = im.shape
    # 调整ROI位置，确保不超出图像边界
    if newx >= diff_xlen / 2 and newx + roiwidth + diff_xlen / 2 < imagecols:
        newx = newx - diff_xlen / 2
    elif newx < diff_xlen / 2:
        newx = 0
    else:
        newx = imagecols - dstlen

    if newy >= diff_ylen / 2 and newy + roiheight + diff_ylen / 2 < imagerows:
        newy = newy - diff_ylen / 2
    elif newy < diff_ylen / 2:
        newy = 0
    else:
        newy = imagerows - dstlen

    # 提取并处理ROI
    roi = im[int(newy):int(newy + dstlen), int(newx):int(newx + dstlen), 0:3]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roiresized = cv2.resize(roi, (testsize, testsize))
    imgblob = data_transforms(roiresized).unsqueeze(0)
    imgblob.requires_grad = False
    predict = F.softmax(net(imgblob), dim=1)  # 使用模型进行预测
    print(predict)
    index = np.argmax(predict.detach().numpy())  # 获取预测结果的索引

    # 显示结果
    im_show = cv2.imread(os.path.join(image_folder_path, imagepath), 1)
    im_h, im_w, im_c = im_show.shape
    pos_x = int(newx + dstlen)
    pos_y = int(newy + dstlen)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(im_show, (int(newx), int(newy)), (int(newx + dstlen), int(newy + dstlen)), (0, 255, 255), 2)
    # 根据预测结果在图像上显示文本
    if index == 0:
        cv2.putText(im_show, 'none', (pos_x, pos_y), font, 1.2, (0, 255, 255), 2)
    elif index == 1:
        cv2.putText(im_show, 'pouting', (pos_x, pos_y), font, 1.2, (0, 255, 255), 2)
    elif index == 2:
        cv2.putText(im_show, 'smile', (pos_x, pos_y), font, 1.2, (0, 255, 255), 2)
    elif index == 3:
        cv2.putText(im_show, 'open', (pos_x, pos_y), font, 1.2, (0, 255, 255), 2)
    cv2.namedWindow('result', 0)
    cv2.imshow('result', im_show)
    cv2.imwrite(os.path.join('results', imagepath), im_show)  # 保存结果图像
    cv2.waitKey(0)  # 等待按键继续
