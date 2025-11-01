import csv
from collections import defaultdict

# Archivos
csv_input  = "p4_results.csv"         # CSV generado por tu script principal
csv_output = "p4_flujo.csv"     # CSV con flujo calculado (una fila por ID)

# Historial de centros y datos de cada track
track_history = defaultdict(list)
track_info = {}  # Guardará info básica del último registro por ID

# Leer CSV original
with open(csv_input, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        track_id = int(row['track_id'])
        x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])

        # Centro de la detección
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        track_history[track_id].append((cx, cy))

        # Guardamos la última fila para mantener info de clase y confianza
        track_info[track_id] = row

# Calcular flujo final por track_id
track_flow = {}
for track_id, centers in track_history.items():
    if len(centers) < 2:
        flujo = "desconocido"
    else:
        cx_inicial, _ = centers[0]
        cx_final, _   = centers[-1]
        if cx_final > cx_inicial:
            flujo = "derecha"
        elif cx_final < cx_inicial:
            flujo = "izquierda"
        else:
            flujo = "estatica"
    track_flow[track_id] = flujo

# Preparar CSV de salida
fieldnames = ['track_id', 'tipo_objeto', 'confianza', 'flujo']

with open(csv_output, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for track_id, row in track_info.items():
        writer.writerow({
            'track_id': track_id,
            'tipo_objeto': row['tipo_objeto'],
            'confianza': row['confianza'],
            'flujo': track_flow.get(track_id, 'desconocido')
        })

print(f"CSV de flujo final generado: {csv_output}")

