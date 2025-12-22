# Menggunakan image Python versi 3.8 sebagai base
FROM python:3.8-slim

# Salin file conda.yaml ke dalam container
COPY conda.yaml /tmp/conda.yaml

# Install dependencies menggunakan Conda
RUN conda env create -f /tmp/conda.yaml

# Salin semua file proyek ke dalam container
COPY . /app

# Atur direktori kerja di dalam container
WORKDIR /app

# Install MLflow dan dependency lainnya
RUN pip install mlflow

# Tentukan perintah yang akan dijalankan saat container dimulai
CMD ["python", "MLProject/modelling.py"]
