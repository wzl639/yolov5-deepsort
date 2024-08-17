import cv2
import numpy as np
import onnxruntime

"""
yolov5onnx 模型推理脚本，后处理使用numpy，模型输出[1,25200,5+classnum]格式（使用yolo5-master将pt模型转换为onnx）
参考链接：https://blog.csdn.net/qq_22487889/article/details/128011883
"""

CLASSES = ['boat', '散货船', '危化品船', '水泥船', '集装箱船', '客船', '滚装船',
                     '工作船', 'text', 'text1']


def preprocess(img, size=(640, 640)):
    """
    yolo数据预处理
    :param img: cv2读取的图片
    :param size: 模型输入尺寸
    :return: 模型输入的np数组
    """
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = image.transpose(2, 0, 1)  # 调整通道顺序
    image = image / 255.0  # 归一化
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)  # 增加批次维度
    return image


# dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
# thresh: 阈值
def nms(dets, thresh):
    """
    非极大值阈值
    :param dets: (n,5)
    :param thresh:
    :return:
    """
    # print(dets)
    # 计算每个框的面积
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)  # (n)
    # 框的置信度倒排的索引 比如[0 7 8 3 4 2 6 5 1]
    scores = dets[:, 4]
    index = scores.argsort()[::-1]  # np.argsort()对某维度从小到大排序
    # NMS过程
    keep = []  # 保存选中框的下标
    while index.size > 0:
        i = index[0]
        keep.append(i)
        #  计算当前框和其他所有框的相交面积
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def postprocess(pre, conf_thres, iou_thres, model_size=(640, 640), img_size=(1920, 1400), label={0:"boat"}):
    """
    yolov5模型输出进行解码阈值、类别和NMS处理后的框，框的坐标还原
    :param pre: np数组 模型预测输出，格式[1,25200,5+classnum] 类别概率和坐标已经进过解码
    :param conf_thres: 物体置信度阈值
    :param iou_thres: NMS阈值
    :param model_size: 模型输入尺寸（h, w）
    :param img_size: 实际图片尺寸（h, w）
    :return: out_boxs: [[x1,y1,x2,y2,pro,cls]..]
    """
    # objectness score置信度过滤
    pre = np.squeeze(pre)  # 删除为1的维度 (25200, 9)
    conf = pre[..., 4] > conf_thres  # […,4]：代表了取最里边一层的第4号,置信度 > conf_thres 的
    box = pre[conf == True]  # 根据objectness score生成(n, 4+1+numclass)，只留下符合要求的框
    # print('box:符合要求的框:', box.shape)
    if len(box) == 0:
        return []
    #  通过argmax获取置信度最大的类别 (n, x,y,w,h,pro,numclass) -> (n, x,y,x,y,pro,cls)
    new_box = []
    all_cls = set()  # 用来记录所有检测到的内别
    for i in range(len(box)):
        cls = np.argmax(box[i][5:])
        box[i][5] = cls
        new_box.append(box[i][:6])  # (x,y,w,h,cls)
        all_cls.add(cls)

    # 坐标形式转换 (n, (x,y,w,h,pro, cls)) -> (n, (x,y,x,y,pro, cls))
    box = np.array(new_box)
    box = xywh2xyxy(box)
    # 框坐标还原到原始图片
    img_scale = np.array([img_size[1] / model_size[1], img_size[0] / model_size[0],
                          img_size[1] / model_size[1], img_size[0] / model_size[0]])
    # 分别对每个类别进行非极大抑制过滤
    out_boxs = []
    for cls in all_cls:
        # 获取当前类别的所有框
        cla_msk = box[:,5] == cls
        curr_cls_box = box[cla_msk==True]
        # 当前类别框NMS
        curr_out_box = nms(curr_cls_box, iou_thres)
        # print(curr_out_box)
        for k in curr_out_box:
            boxx, pre, cls = curr_cls_box[k][:4]*img_scale, curr_cls_box[k][4], int(curr_cls_box[k][5])
            out_boxs.append([boxx[0], boxx[1], boxx[2], boxx[3], pre, label.get(cls, cls)])
    return out_boxs


def detect(image):
    # 读图预处理
    # image_path = 'test.jpg'
    # image = cv2.imread(image_path)
    h, w, _ = image.shape
    input_np = preprocess(image)

    # onnx进行推理
    # onnx_model_path = './detector/boat_det_siyang.onnx'
    onnx_model_path = './detector/car_yolov5x.onnx'
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    pre = session.run(None, {input_name: input_np})[0]  # <class 'numpy.ndarray'> (1, 25200, 15)
    # print(type(pre), pre.shape)

    # 推理结果解码后处理
    outbox = postprocess(pre, 0.4, 0.3, (640, 640), (h, w))
    return outbox


if __name__ == '__main__':
    # 读图预处理
    image_path = 'car_test.jpg'
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    input_np = preprocess(image)

    # onnx进行推理
    onnx_model_path = 'car_yolov5x.onnx'
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    pre = session.run(None, {input_name: input_np})[0]  # <class 'numpy.ndarray'> (1, 25200, 15)
    # print(type(pre), pre.shape)

    # 推理结果解码后处理
    outbox = postprocess(pre, 0.8, 0.3, (640, 640), (h, w))
    print(outbox)
    # 结果可视化显示
    for x1, y1, x2, y2, score, cl in outbox:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, p1, p2, (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(cl, score),
                    p1, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 1000, 600)
    cv2.imshow('show', image)
    cv2.waitKey(5000)
