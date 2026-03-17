import joblib

model = joblib.load("models\\asl_model.pkl")

def predict(data):
    return model.predict(data)[0]