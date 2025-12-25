import time
import random
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metric persis seperti di patokan
REQUEST_COUNT = Counter('ml_request_total', 'Total Request', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('ml_request_latency_seconds', 'Latency', ['endpoint'])
PREDICTION_VALUE = Gauge('ml_prediction_value', 'Nilai Prediksi')

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    # Simulasi prediksi
    price = 20.0 + random.uniform(-5, 5)
    
    REQUEST_COUNT.labels(endpoint='/predict', status='success').inc()
    REQUEST_LATENCY.labels(endpoint='/predict').observe(time.time() - start_time)
    PREDICTION_VALUE.set(price)
    
    return jsonify({"price": price})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)