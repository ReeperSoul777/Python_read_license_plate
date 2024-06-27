import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
import easyocr
import csv


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
sys.path.append(str(Path(__file__).resolve().parent / '../yolov9'))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# Inicjalizacja easyocr
reader = easyocr.Reader(['pl'])
confidence_threshold = 0.6 #próg odrzucania wyników
min_length = 6 #minimalna ilość znaków w tablicy rejestracyjnej

RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"

def deskew_image(image):
    '''
    prostowanie perspektywy
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # wykrywanie krawędzi
    edges = cv2.Canny(gray, 0, 200)

    # szukanie konturu
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:

            plate_contour = approx
            break

    # narożniki
    pts = plate_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # sortowanie: górny lewy, górny prawy, dolny prawy, dolny lewy
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    # Oblicz szerokość i wysokość
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Nowe wymiary obrazu
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # transformacja perspektywiczna
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return(warped)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def smooth_edges(image):
    smoothed = cv2.bilateralFilter(image, 9, 75, 75)
    return smoothed

def read_license_plate(image, thresholding, enhance, debug, save_results, save_rejections, frame_number):
    """
    Odczytuje tablicę rejestracyjną ze zdjęcia.

    Parametry:
    - image_path (str): Ścieżka do obrazu, który ma być przetworzony.
    - thresholding (str): Metoda progowania obrazu.
                 - 'adaptive': Adaptacyjne progowanie.
                 - 'otsu': Progowanie Otsu.
                 - 'none': Bez progowania.
    - enhance (str): Metoda ulepszania obrazu.
                 - 'sharpen': Wyostrzenie obrazu.
                 - 'smooth': Wygładzenie krawędzi.
                 - 'none': Bez ulepszania.
    - debug (bool): Jeśli True, pokazuje obrazy na różnych etapach przetwarzania.

    Zwraca:
    - str: Odczytany tekst z tablicy rejestracyjnej.
    """
    # Sprawdzenie, czy obraz jest ścieżką, czy już załadowanym obrazem
    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"Obraz nie został wczytany. Sprawdź ścieżkę: {image}")
        image = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Obraz nie został wczytany poprawnie: {image}")

    if debug:
        print('Orginalny obraz')
        cv2.imshow('Original Image', image)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
        

    if debug: print('Konwersja na skale szarości')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if debug:
        cv2.imshow('Grayscale Image', gray)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    # Ulepszanie obrazu
    if enhance == 'sharpen':
        if debug: print('Wyostrzanie krawędzi')
        gray = sharpen_image(gray)
    elif enhance == 'smooth':
        if debug: print('Wygładzanie krawędzi')
        gray = smooth_edges(gray)
        

    if debug:
        print ('ulepszony obraz')
        cv2.imshow('Enhanced Image', gray)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    # Wybór metody progowania
    if thresholding == 'adaptive':
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        if debug: print('adaptive')
    elif thresholding == 'otsu':
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if debug: print('otsu')

    if debug:
        cv2.imshow('Thresholded Image', gray)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    #ocr na pliku, dodany allowlist w celu ograniczenia ilości znaków
    results = reader.readtext(
        gray,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        text_threshold=0.7,
        adjust_contrast=0.7, #współczynnik regulacji kontrastu
        contrast_ths=0.5, #próg kontrastu używany do przetwarzania obrazu
        link_threshold=0.15, #próg łączenia wiersza
        paragraph=False
)

    filtered_text = []
    for result in results:
        text, confidence = result[1], result[2]
        if confidence >= confidence_threshold:
            if len(text) >= min_length:
                filtered_text.append(text)
                print(f"{GREEN}Odczytane numery tablic:  {text}{RESET}")
                # Save recognized license plate to CSV
                if save_results:
                    save_results_to_csv(frame_number, text)

            else:
                reason = f"Rejected due to short length (length: {len(text)})"
                if debug: print(f"{YELLOW}Rejected due to short length (length: {len(text)}){RESET}")
                # Save rejection to CSV
                if save_rejections:
                    save_rejections_to_csv(frame_number, text, reason)
        else:
            reason = f"Rejected due to low confidence (confidence: {confidence:.2f})"
            if debug: print(f"{YELLOW}Rejected due to low confidence (confidence: {confidence:.2f}){RESET}")
            # Save rejection to CSV
            if save_rejections:
                save_rejections_to_csv(frame_number, text, reason)

    text = " ".join(filtered_text)


    if debug:
        print(f"OCR Results: {results}")

def save_results_to_csv(frame_number, recognized_text):
    csv_file = "recognized_license_plates.csv"
    fieldnames = ['Frame Number', 'License Plate']
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({'Frame Number': frame_number, 'License Plate': recognized_text})


def save_rejections_to_csv(frame_number, rejected_text, reason):
    csv_file = "rejected_license_plates.csv"
    fieldnames = ['Frame Number', 'Rejected License Plate', 'Reason']
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()


    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({'Frame Number': frame_number, 'Rejected License Plate': rejected_text, 'Reason': reason})


def run(
        read_license=True, # detect license number on frames
        weights='yolov9-s4/weights/best.pt',  # model path or triton URL
        source='./video4.mp4',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='dectected',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        thresholding='none',
        enhance='sharpen',
        debug=False,
        save_results=True,
        save_rejections=True,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    iplk = 0
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            pred = pred[0][1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        iplk += 1
                        crop_img = save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}{iplk}.jpgess', BGR=True)
                    if read_license:
                        # Przetwarzanie wyciętego obrazu tablicy rejestracyjnej
                        #read_license_plate(str(save_dir / 'crops' / names[c] / f'{p.stem}{iplk}.jpg'), thresholding='none', enhance='sharpen', debug=False)
                        crop_img = save_one_box(xyxy, imc, BGR=True, save=False)
                        read_license_plate(crop_img, thresholding, enhance, debug, save_results, save_rejections, frame_number=frame)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
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
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def main():
    run()

if __name__ == "__main__":
    main()
