#-*-coding:UTF-8 -*-

import matplotlib.pyplot as plt # plotting
from torchvision import transforms as T
import torchvision




import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    
    second_stage_classify=opt.second_stage_classify
    second_stage_classifier=opt.second_stage_classifier
    
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier chuhr modified here
    if second_stage_classify:
#         Second_stage_classes=['Spotted Dove',
#              'Eurasian Tree Sparrow',
#              'Black-winged Kite',
#              'Barn Swallow',
#              'Oriental Magpie-Robin',
#              'Common Myna',
#              'Silky stanling',
#              'Common Stonechat',
#              'Paddyfield Pipit',
#              'Black-collared Starling',
#              'Red-rumped Swallow',
#              'Scaly-breasted Munia',
#              'Ardeola bacchus',
#              'Chinese Sparrowhawk',
#              'Red-whiskered Bulbul',
#              'Rock Sparrow',
#              'Black-faced Laughingthrush',
#              'Common Tailorbird',
#              'Japanese White-eye',
#              'Bank Myna',
#              'Russet Sparrow',
#              'Eurasian Blackbird',
#              'Striated Prinia',
#              'Light-vented Bulbul',
#              'Japanese Sparrowhawk',
#              'White Wagtail',
#              'Eurasian Sparrowhawk',
#              'Sooty-headed Bulbul',
#              'House Sparrow',
#              'Chestnut Bulbul',
#              'White-vented Myna',
#              'Hill Myna',
#              'Crested Myna',
#              'White-rumped Munia',
#              'Golden-crested Myna',
#              'Spanish Sparrow',
#              'Dusky Warbler',
#              'Collared Myna',
#              'Java Sparrow',
#              'Saxaul Sparrow']
        Second_stage_classes=['珠颈斑鸠',
             '[树]麻雀',
             '黑翅鸢',
             '家燕',
             '鹊鸲',
             '家八哥',
             '丝光椋鸟',
             '黑喉石鵖',
             '田鹨',
             '黑领椋鸟',
             '金腰燕',
             '斑文鸟',
             '池鹭',
             '赤腹鹰',
             '红耳鹎',
             '石雀',
             '黑顶噪鹛',
             '长尾缝叶莺',
             '暗绿绣眼鸟',
             '灰背岸八哥',
             '山麻雀',
             '乌鸫',
             '山鹪莺',
             '白头鹎',
             '日本松雀鹰',
             '白鹡鸰',
             '雀鹰',
             '白喉红臀鹎',
             '家麻雀',
             '栗背短脚鹎',
             '林八哥',
             '鹩哥',
             '八哥',
             '白腰文鸟',
             '金冠树八哥',
             '黑胸麻雀',
             '褐柳莺',
             '白领八哥',
             '禾雀',
             '黑顶麻雀']




        if second_stage_classifier=="resnet50":
            modelc =torchvision.models.resnet50()
            modelc.load_state_dict(torch.load('classifier_weights/resnet50.pt', map_location=device))  # load weights
        modelc.to(device).eval()
        print("successfully load resnet 50 trained by CHR")

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    #储黄瑞在这里更改
#                     print("\n   bbox 信息")
#                     print(im0.shape)
                    #这边是boundingbox
                    x1,y1,x2,y2=xyxy
#                     print(x1)
#                     print(x2)
#                     print(y1)
#                     print(y2)
                    #这里剪出来那个鸟，颜色的输入不是RGB，BGR吧？
                    cur_bird=im0[int(y1):int(y2),int(x1):int(x2),:]
                    from PIL import Image
                    im = Image.fromarray(cv2.cvtColor(cur_bird,cv2.COLOR_BGR2RGB)) 
                    #im = Image.fromarray(cur_bird)
                    transform = T.Compose([
                    T.Resize([224, 224]),
                    T.ToTensor(),
                    T.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    im =transform(im).to(device)          
                    #把处理好的图片放入分类器
                    output=modelc(im.unsqueeze(0))
                    pred = output.argmax(dim=1, keepdim=True)[0][0].cpu()
#                     print("置信度")
#                     print(output[0][pred])
#                     print("标签")
#                     print(pred)
#                     print("完成了！")
                  
                   
                    
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        if second_stage_classify:
                            print("标注具体鸟类类别中")
                            label2='%s %.2f' % (Second_stage_classes[int(pred)], conf)
                            print(label2)
                            #plot_one_box(xyxy, im0, label=label2, color=colors[int(cls)], line_thickness=3)
                            im0=plot_one_box(xyxy, im0, label=label2, color=colors[int(cls)], line_thickness=3)
                        else:
                            label = '%s %.2f' % (names[int(cls)], conf)
                            im0=plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--second_stage_classify', type=bool, default=True, help='是否细化检测鸟类的品种')
    parser.add_argument('--second_stage_classifier', type=str, default="resnet50", help='检测鸟类的品种的分类器')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
