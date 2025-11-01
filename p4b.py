from ultralytics import YOLO
import cv2
import easyocr
from collections import defaultdict

vehicle_model = YOLO('yolo11n.pt')                  # personas y vehículos
plate_model   = YOLO('yolo_runs/plates_detection/weights/best.pt')  # placas
ocr_reader = easyocr.Reader(['es'])

vid = cv2.VideoCapture("C0142.mp4")
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))

# Crear writer para guardar el vídeo
out = cv2.VideoWriter("resultado.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

track_history = defaultdict(lambda: [])
classNames = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
vehicle_classes = [0,1,2,3,4,5]

while True:
    ret, frame = vid.read()
    if not ret: break

    # Tracking de personas y vehículos
    results = vehicle_model.track(frame, persist=True, classes=vehicle_classes)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            # Dibujar contenedor
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{track_id}-{classNames[cls]}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            # Si es un vehículo, exceptuando motos y bicicicletas, aplicar detector de matrícula
            if cls in [2,4,5]:
                vehicle_crop = frame[y1:y2, x1:x2]
                plate_results = plate_model(vehicle_crop)

                for p in plate_results:
                    for pb in p.boxes:
                        px1, py1, px2, py2 = map(int, pb.xyxy[0])
                        cv2.rectangle(vehicle_crop, (px1,py1),(px2,py2),(0,0,255),2)

                        # Añadir OCR - Detección de texto de matrícula

    # Guardar frame en vídeo
    out.write(frame)

vid.release()
out.release()
print("Vídeo resultado guardado como 'resultado.mp4'")
