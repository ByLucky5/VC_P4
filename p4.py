import cv2
import csv
from ultralytics import YOLO

# Configuración
video_input = "C0142.mp4"
video_output = "p4_output.mp4"
csv_output   = "p4_results.csv"

general_model = YOLO('yolo11n.pt')  # personas y vehículos
plate_model   = YOLO('yolo_runs/plates_detection/weights/best.pt')  # placas

classNames = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
general_classes = [0,1,2,3,4,5]  # solo personas y vehículos

# Preparar captura y salida
vid = cv2.VideoCapture(video_input)
width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = vid.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

# CSV
csv_file = open(csv_output, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'tipo_objeto', 'confianza', 'track_id',
                     'x1','y1','x2','y2', 'plate_conf', 'mx1','my1','mx2','my2'])

frame_count = 0

# Procesamiento fotograma a fotograma
while True:
    ret, frame = vid.read()
    if not ret: 
        break

    frame_count += 1

    # Tracking de personas y vehículos
    results = general_model.track(frame, persist=True, classes=general_classes)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            conf = float(box.conf[0])

            # Anonimizar: desenfoque si es persona
            if cls == 0:  # persona
                person_roi = frame[y1:y2, x1:x2]
                blur = cv2.GaussianBlur(person_roi, (25,25), 30)
                frame[y1:y2, x1:x2] = blur

            # Dibujar contenedor
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{track_id}-{classNames[cls]}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            # Si es un vehículo, exceptuando motos y bicicicletas, aplicar detector de matrícula
            plate_conf = ''
            mx1 = my1 = mx2 = my2 = ''
            if cls in [2,4,5]:
                vehicle_crop = frame[y1:y2, x1:x2]
                plate_results = plate_model(vehicle_crop)
                for p in plate_results:
                    for pb in p.boxes:
                        px1, py1, px2, py2 = map(int, pb.xyxy[0])
                        cv2.rectangle(vehicle_crop, (px1,py1),(px2,py2),(0,0,255),2)
                        plate_conf = float(pb.conf[0])
                        mx1, my1 = x1+px1, y1+py1
                        mx2, my2 = x1+px2, y1+py2

                # Desenfoque de matrícula para anonimizar
                if mx1 != '' and my1 != '':
                    plate_roi = frame[my1:my2, mx1:mx2]
                    plate_blur = cv2.GaussianBlur(plate_roi, (25,25), 30)
                    frame[my1:my2, mx1:mx2] = plate_blur

            # Guardar info en CSV
            csv_writer.writerow([frame_count, classNames[cls], conf, track_id,
                                 x1, y1, x2, y2, plate_conf, mx1, my1, mx2, my2])

    out_vid.write(frame)

# ========================
# Liberar recursos
# ========================
vid.release()
out_vid.release()
csv_file.close()
