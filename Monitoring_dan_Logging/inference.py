import requests
import time

print("Running traffic...")
while True:
    try:
        requests.post("http://localhost:5001/predict", json={})
        print(".", end="", flush=True)
    except:
        pass
    time.sleep(1)