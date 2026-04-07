from model import predict

def run_inference(text):
    pred, conf, prob = predict(text)
    
    return {
        "prediction": int(pred),
        "confidence": float(conf),
        "probabilities": prob
    }

if __name__ == "__main__":
    sample = "Aliens landed on Earth"
    print(run_inference(sample))
