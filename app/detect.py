import cv2
import torch
from ultralytics import YOLO
import numpy as np
import logging
import time
import threading
from enum import Enum, auto
import json
import requests

# Configuração do logger
LOG_FILE = "detections.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Definição da máquina de estados 
class State(Enum):
    VERIFY_CONNECTION     = auto()
    CAPTURE_PROCESS       = auto()
    CHECK_FALSE_POSITIVE  = auto()
    SEND_ALERT            = auto()
    VERIFY_SEND           = auto()

# Stub de envio HTTP
class FakeResponse:
    def __init__(self, status_code): self.status_code = status_code

def fake_http_post(url, payload):
    # simula sucesso ou falha
    print(f"Enviando alert para {url}: {json.dumps(payload)}")
   # return FakeResponse(200)
    
    resp = requests.post(url, json=payload, timeout=5)
    return resp

# Funções de desenho e detecção 
def draw_bounding_box(frame, box, class_name, confidence):
    x1, y1, x2, y2 = [int(c) for c in box]
    label = f"{class_name} {confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def detect_fire_and_smoke(model, frame, conf_thresh=0.001):
    results = model.predict(frame, verbose=False)[0]
    fire_detected = False

    for i, conf in enumerate(results.boxes.conf):
        if conf < conf_thresh:
            continue
        cls_id = int(results.boxes.cls[i])
        name = results.names[cls_id].lower()
        box = results.boxes.xyxy[i].tolist()
        draw_bounding_box(frame, box, name, conf.item())
        if name == "fire":
            fire_detected = True

    return fire_detected, frame

# Worker de cada câmera com máquina de estados 
def camera_worker(model, url, name, shared_frames, index, lock):
    cap = None
    state = State.VERIFY_CONNECTION
    frames_with_fire = 0
    FIRE_CONFIRMATION_THRESHOLD = 3

    while True:
        if state == State.VERIFY_CONNECTION:
            if cap is None:
                cap = cv2.VideoCapture(url)
            if cap.isOpened():
                logging.info(f"Conexão OK em '{name}'")
                state = State.CAPTURE_PROCESS
            else:
                logging.error(f"Falha na conexão de '{name}', tentando novamente...")
                time.sleep(1)
                continue

        elif state == State.CAPTURE_PROCESS:
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Leitura falhou em '{name}', reiniciando captura...")
                cap.release()
                cap = None
                state = State.VERIFY_CONNECTION
                continue

            frame = cv2.resize(frame, (320, 240))
            fire_detected, annotated = detect_fire_and_smoke(model, frame)

            with lock:
                shared_frames[index] = annotated

            if fire_detected:
                frames_with_fire += 1
                state = State.CHECK_FALSE_POSITIVE
            else:
                frames_with_fire = 0
              
        elif state == State.CHECK_FALSE_POSITIVE:
            if frames_with_fire >= FIRE_CONFIRMATION_THRESHOLD:
                state = State.SEND_ALERT
            else:
                state = State.CAPTURE_PROCESS

        elif state == State.SEND_ALERT:
            alert = {
                "camera": name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "confirmed_frames": frames_with_fire
            }
            try:
                response = fake_http_post("http://meu_servidor/alert", alert)
                success = (response.status_code == 200)
            except Exception as e:
                logging.error(f"Erro ao enviar JSON de '{name}': {e}")
                success = False

            state = State.VERIFY_SEND if success else State.SEND_ALERT

        elif state == State.VERIFY_SEND:
            logging.info(f"🔥🚨 Alerta enviado de '{name}' com {frames_with_fire} frames confirmados")
            frames_with_fire = 0
            state = State.CAPTURE_PROCESS

# Gerenciamento das threads e janelas 
def play_three_parallel(model, video_urls):
    shared_frames = [
        np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in video_urls
    ]
    lock = threading.Lock()
    threads = []

    for i, url in enumerate(video_urls):
        t = threading.Thread(
            target=camera_worker,
            args=(model, url, f"Cam {i}", shared_frames, i, lock),
            daemon=True
        )
        t.start()
        threads.append(t)

    cv2.namedWindow("Videos", cv2.WINDOW_NORMAL)
    while True:
        with lock:
            combined = np.hstack(shared_frames)
        cv2.imshow("Videos", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
  
# Função principal 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("runs/detect/train3/weights/best.pt").to(device)

    base = "http://localhost:5000/video_feed?video_index="
    video_urls = [base + str(i) for i in range(3)]

    play_three_parallel(model, video_urls)

if __name__ == "__main__":
    main()
