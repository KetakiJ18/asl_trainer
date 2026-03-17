from collections import deque, Counter

prediction_buffer = deque(maxlen=10)

def smooth_prediction(pred):
    prediction_buffer.append(pred)

    if len(prediction_buffer) == 10:
        return Counter(prediction_buffer).most_common(1)[0][0]

    return ""