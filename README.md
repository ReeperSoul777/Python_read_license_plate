
# License Plate Detection and Recognition 

This project aims to detect and recognize vehicle license plates from video streams using YOLOv9 and EasyOCR. The program captures frames from a video stream, detects license plates using a pre-trained YOLOv9 model, and performs Optical Character Recognition (OCR) on the detected plates.

## Features
- Real-time license plate detection from video streams.
- Optical Character Recognition (OCR) for detected license plates.
- Configurable to use pre-trained YOLOv9 models or self trained models

## Installation

Follow these steps to set up the project:

### Clone the Repository
```bash
git clone https://github.com/ReeperSoul777/Python_read_license_plate.git
```

### Clone the YOLOv9 and install Dependencies
```bash
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9
pip install -r requirements.txt
```


### Install EasyOCR
Ensure you have Paddle OCR installed on your system. You can download it from 
```bash
pip install easyocr
```

## Usage

Run GUI and specify input

## File Structure
- `yolov9`: YOLOv9 model
- `yolov9-s4`: pretrained weights



## License
This project is licensed under the AGPL-3.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [openalpr](https://github.com/openalpr/) - training data

```
