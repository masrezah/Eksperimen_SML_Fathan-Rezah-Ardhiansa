from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
import requests

# Metrik untuk Skilled (Minimal 3)
REQUEST_COUNT = Counter('request_count', 'Total Request')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Latency')
PREDICTION_VALUE = Gauge('prediction_value', 'Nilai Prediksi')

URL = "http://127.0.0.1:5001/invocations"

def process_request():
    start = time.time()
    try:
        # Kirim data dummy ke server serving
        resp = requests.post(URL, json={"data": []})
        
        # Hitung durasi
        latency = time.time() - start
        
        # Update Metrik
        REQUEST_COUNT.inc()
        REQUEST_LATENCY.observe(latency)
        
        if resp.status_code == 200:
            val = resp.json()['predictions'][0]
            PREDICTION_VALUE.set(val)
            print(f"Sukses! Prediksi: {val:.2f} | Latency: {latency:.2f}s")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # Jalan di port 8000
    start_http_server(8000)
    print("Exporter jalan di port 8000...")
    
    while True:
        process_request()
        time.sleep(2)