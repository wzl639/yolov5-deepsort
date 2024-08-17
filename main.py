import cv2
import numpy as np
from deep_sort import DeepSort
from detector.yolov5_onnx import Yolov5Onnx

"""
目标追踪示例，示例中使用检测是船舶检测模型，由于目标比较大，追踪过程检测框和目标匹配不是准确
"""


def main(rid_model_path, detect_model_path, video_path):

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1920, 1080))

    # 初始化追踪器
    deepsort = DeepSort(model_path=rid_model_path,
                        max_dist=0.2, min_confidence=0.3,
                        nms_max_overlap=0.5, max_iou_distance=0.7,
                        max_age=70, n_init=3, nn_budget=100, use_cuda=True)

    # 初始化检测器
    model = Yolov5Onnx(detect_model_path)

    # 读视频流，检测追踪，可视化
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    while True:
        ret, frame = cap.read()
        cnt += 1
        if not ret:
            break
        # if cnt % 1 != 0:
        #     continue
        frame = cv2.resize(frame, (1920, 1080))
        # 当前帧做检测
        det_boxes = model.detect(frame)
        if len(det_boxes) == 0:
            continue
        # print(det_boxes)
        # 结果可视化显示
        bbox_xywh = []
        confs = []
        labels = []
        for detbox in det_boxes:
            x1, y1, x2, y2 = detbox[:4]
            obj = [int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1]
            bbox_xywh.append(obj)
            confs.append(detbox[4])
            labels.append(detbox[5])
        xywhs = np.array(bbox_xywh)
        confss = np.array(confs)

        # 追踪
        outputs = deepsort.update(xywhs, confss, frame)  # [[   0   77 1919  790    3]]
        # print(outputs)
        # 可视化追踪结果
        for (x1, y1, x2, y2, track_id) in outputs:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.putText(frame, "track_id:" + str(track_id), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness=3, lineType=cv2.LINE_8)

        cv2.namedWindow('show', 0)
        cv2.resizeWindow('show', 1000, 600)
        cv2.imshow('show', frame)
        cv2.waitKey(1)

        # out.write(frame)
    # 释放视频捕获对象和关闭窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    rid_model_path = "./deep_sort/deep/checkpoint/ckpt.t7"  # ReId 模型路径
    detect_model_path = "./detector/boat_det_siyang.onnx"  # 检测模型路径
    video_path = "test.mp4"  # 视频路径
    main(rid_model_path, detect_model_path, video_path)