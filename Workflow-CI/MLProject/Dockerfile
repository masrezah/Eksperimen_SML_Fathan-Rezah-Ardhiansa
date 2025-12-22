# Menggunakan image dasar Python
FROM python:3.8-slim

# Menginstal dependencies
COPY conda.yaml /tmp/conda.yaml
RUN conda env create -f /tmp/conda.yaml

# Menyalin file proyek ke dalam container
COPY . /app
WORKDIR /app

# Instal MLflow jika belum ada
RUN pip install mlflow

# Default command
CMD ["python", "MLProject/modelling.py"]
