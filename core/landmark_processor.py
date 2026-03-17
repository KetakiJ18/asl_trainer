import numpy as np

def process_landmarks(hand_landmarks):
    coords = []

    for point in hand_landmarks.landmark:
        coords.append([point.x, point.y, point.z])

    coords = np.array(coords)

    wrist = coords[0]
    coords = coords - wrist

    max_value = np.max(np.abs(coords))

    if max_value != 0:
        coords = coords / max_value

    return coords.flatten().reshape(1, -1)