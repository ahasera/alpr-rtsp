import cv2
import torch
import threading
import time
from flask import Flask, Response, render_template
import queue
import easyocr
import paddleocr
from ultralytics import YOLO
import numpy as np
import re
import argparse

app = Flask(__name__)

'''
#######################
Parsing OCR engine args 
#######################
'''

parser = argparse.ArgumentParser(description='Choose OCR engine')
parser.add_argument('--ocr', choices=['easyocr', 'paddleocr'], default='easyocr', help='Choose OCR engine (easyocr or paddleocr)')
args = parser.parse_args()

'''
########################################################################
Initialisation of MS-COCO pretained YOLOv8 and LP detection custom model
########################################################################
'''

torch.backends.cudnn.benchmark = True  
model_vehicle = YOLO('models/yolov8n.pt').to('cuda')
model_plate = YOLO('models/best.pt').to('cuda')

'''
############################################
Initialisation of OCR engines depend on args
############################################   
'''
if args.ocr == 'easyocr':
    reader_easyocr = easyocr.Reader(['en'], gpu=True)
    ocr_engine = reader_easyocr
elif args.ocr == 'paddleocr':
    ocr_paddle = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    ocr_engine = ocr_paddle

frame_queue = queue.Queue(maxsize=10)
rtsp_url = 'rtsp://192.168.1.19:8554/webcam' # change this to the URL of your RTSP server 

'''
########################
Patterns for EUROPEAN LP 
########################
'''

EUROPEAN_PATTERNS = {
    'FR': r'^(?:[A-Z]{2}-\d{3}-[A-Z]{2}|\d{2,4}\s?[A-Z]{2,3}\s?\d{2,4})$',  # France
    'DE': r'^[A-Z]{1,3}-[A-Z]{1,2}\s?\d{1,4}[EH]?$',  # Germany
    'ES': r'^(\d{4}[A-Z]{3}|[A-Z]{1,2}\d{4}[A-Z]{2,3})$',  # Spain
    'IT': r'^[A-Z]{2}\s?\d{3}\s?[A-Z]{2}$',  # Italy
    'GB': r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$',  # Great-Britain
    'NL': r'^[A-Z]{2}-\d{3}-[A-Z]$',  # Netherlands
    'BE': r'^(1-[A-Z]{3}-\d{3}|\d-[A-Z]{3}-\d{3})$',  # Belgium
    'PL': r'^[A-Z]{2,3}\s?\d{4,5}$',  # Poland
    'SE': r'^[A-Z]{3}\s?\d{3}$',  # Sweden
    'NO': r'^[A-Z]{2}\s?\d{5}$',  # Norway
    'FI': r'^[A-Z]{3}-\d{3}$',  # Finland
    'DK': r'^[A-Z]{2}\s?\d{2}\s?\d{3}$',  # Denmark
    'CH': r'^[A-Z]{2}\s?\d{1,6}$',  # Switzerland
    'AT': r'^[A-Z]{1,2}\s?\d{1,5}[A-Z]$',  # Austria
    'PT': r'^[A-Z]{2}-\d{2}-[A-Z]{2}$',  # Portugal
    'EU': r'^[A-Z0-9]{2,4}[-\s]?[A-Z0-9]{1,4}[-\s]?[A-Z0-9]{1,4}$'  # Generic European plate
}

def capture_frames(rtsp_url):
    while True:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"Error: Unable to open video stream from {rtsp_url}")
            time.sleep(5)
            continue
        
        while True:
            success, frame = cap.read()
            if success:
                if not frame_queue.full():
                    frame_queue.put(frame)
                else:
                    frame_queue.get()
                    frame_queue.put(frame)
            else:
                print(f"Error: Failed to capture video stream from {rtsp_url}")
                break
            
            time.sleep(0.03)
        
        cap.release()
        time.sleep(1)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    kernel = np.ones((2,2),np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations = 1)
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)

def post_process_ocr(text):
    cleaned_text = re.sub(r'[^A-Z0-9\-\s]', '', text.upper())
    for country, pattern in EUROPEAN_PATTERNS.items():
        if re.match(pattern, cleaned_text.replace(" ", "")):
            return cleaned_text, country
    return cleaned_text, "Unknown"

def perform_ocr(plate, ocr_engine):
    detections = []  

    if ocr_engine == 'easyocr':
        detections = reader_easyocr.readtext(plate)
    elif ocr_engine == 'paddleocr':
        result = ocr_paddle.ocr(plate)
        if result and isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and len(result[0]) > 0:
            detections = [(None, result[0][0][1][0], result[0][0][1][1] if len(result[0][0][1]) > 1 else 0.0)]
    else:
        raise ValueError(f"OCR engine '{ocr_engine}' not supported.")
    
    return detections

def process_frames(ocr_engine='easyocr'):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Vehicle detection
            results_vehicle = model_vehicle(frame)
            for result in results_vehicle:
                for bbox in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = bbox
                    if int(class_id) == 2:  # Assuming class 2 is for cars
                        vehicle = frame[int(y1):int(y2), int(x1):int(x2)]

                        # Plate detection
                        results_plate = model_plate(vehicle)
                        for result_plate in results_plate:
                            for bbox_plate in result_plate.boxes.data.tolist():
                                px1, py1, px2, py2, pscore, pclass_id = bbox_plate
                                plate = vehicle[int(py1):int(py2), int(px1):int(px2)]
                                
                                # OCR on the detected plate
                                detections = perform_ocr(plate, ocr_engine)
                                if len(detections) > 0:
                                    for bbox, text, conf in detections:
                                        processed_text, country = post_process_ocr(text)
                                        cv2.putText(frame, f"{processed_text} ({country})", (int(x1 + px1), int(y1 + py1) - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                        cv2.rectangle(frame, (int(x1 + px1), int(y1 + py1)),
                                                      (int(x1 + px2), int(y1 + py2)), (0, 255, 0), 2)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(ocr_engine='paddleocr'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=capture_frames, args=(rtsp_url,), daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
