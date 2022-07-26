import argparse
import os
import sys
import time
from pathlib import Path
import cv2.aruco as aruco

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y - text_h - 15), text_color_bg, -1)
    cv2.putText(img, text, (x, y - font_scale - 10), font, font_scale, text_color, font_thickness)

def cv2_imshow(image):
    plt.figure(dpi=200)
    mode = len(np.shape(image))
    if mode==3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif mode==2:
        plt.imshow(image, cmap='gray')
    else:
        print('Unsuported image size')
        raise
    plt.xticks([]), plt.yticks([])
    plt.axis('off')

if __name__ == "__main__":
    start_time = time.time()

    source = './our_code/data/video.mp4'


    # Directories
    project = ROOT / 'runs/detect'
    name = 'exp'

    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    (save_dir / 'labels' if False else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device()
    model = DetectMultiBackend(ROOT / 'yolov5s.pt', device=device, dnn=False, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # ArUco marker set up
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)  # Use 5x5 dictionary to find markers
    parameters = aruco.DetectorParameters_create()  # Marker detection parameters

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    since_marker_detect = 0
    since_empty_marker = 0
    corners = []
    for num_frame, (path, im, im0s, vid_cap, s) in enumerate(dataset):


        flag_overlap = False
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Detect ArUco Marker here
        if since_marker_detect <= 0 and since_empty_marker <= 3:
            corners, ids, rejected = cv2.aruco.detectMarkers(image=im0s, dictionary=aruco_dict, parameters=parameters)
            if len(corners) > 2:
                since_marker_detect = 15
                since_empty_marker = 0
            if len(corners) == 0:
                since_empty_marker += 1
        if since_empty_marker > 15:
            since_empty_marker = 0
        since_marker_detect -= 1
        since_empty_marker += 1
        color = (0, 0, 255)
        x_coors = np.array([])
        y_coors = np.array([])
        for corner in corners:
            x_coors = np.append(x_coors, corner.T[0])
            y_coors = np.append(y_coors, corner.T[1])
        if x_coors.any() or y_coors.any():
            x_min = np.min(x_coors)
            y_min = np.min(y_coors)
            x_max = np.max(x_coors)
            y_max = np.max(y_coors)
            cv2.rectangle(im0s, (int(x_max), int(y_max)), (int(x_min), int(y_min)), color, 2)
            org = (int(x_min), int(y_min))
            draw_text(im0s, "Landing Pad", font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, pos=org, text_color_bg=color,
                      text_color=(255, 255, 255))

        # NMS
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh  # for save_crop
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # cv2.imwrite('a.jpg', im0)

                # Write results
                for *xyxy, conf, cls in reversed(det): # Add bbox to image
                    c = int(cls)  # integer class
                    label =  f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    if len(corners) != 0 and flag_overlap == False:
                        if (x_min < xyxy[0] < x_max or x_min < xyxy[2] < x_max) and (
                                y_min < xyxy[1] < y_max or y_min < xyxy[3] < y_max):
                            flag_overlap = True
                        if (xyxy[0] < x_min < xyxy[2] or xyxy[0] < x_max < xyxy[2]) and (
                                xyxy[1] < y_min < xyxy[3] or xyxy[1] < y_max < xyxy[3]):
                            flag_overlap = True

            # Stream results
            im0 = annotator.result()

            # Draw the overlap
            if flag_overlap == True:
                draw_text(im0, "Overlap", font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, pos=(100, 100),
                          text_color_bg=color,
                          text_color=(255, 255, 255))

            # Save results (image with detections)
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    end_time = time.time()
    time_cost = end_time - start_time
    print('The total time to process the video is: {:.4f} seconds.'.format(time_cost))