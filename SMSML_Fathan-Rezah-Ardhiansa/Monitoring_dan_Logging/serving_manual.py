from flask import Flask, request, jsonify
import random
import time

app = Flask(__name__)

@app.route('/invocations', methods=['POST'])
def predict():
    # Simulasi latency (waktu proses)
    time.sleep(random.uniform(0.1, 0.5))
    
    # Simulasi hasil prediksi (Harga rumah random 15-35)
    prediction = random.uniform(15.0, 35.0)
    
    return jsonify({"predictions": [prediction]})

if __name__ == '__main__':
    print("Server Serving berjalan di Port 5001...")
    app.run(host='0.0.0.0', port=5001)