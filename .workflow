name: Data Preprocessing dan Analisis

on:
  push:
    branches:
      - main  # Pastikan nama branch sesuai dengan yang Anda gunakan

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'  # Ganti sesuai dengan versi Python yang Anda gunakan

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt  # Pastikan ada file requirements.txt

      - name: Run preprocessing script
        run: |
          python automate_Nama-siswa.py  # Menjalankan skrip Python Anda

      - name: Visualisasi Data
        script: |
          import matplotlib.pyplot as plt
          import seaborn as sns
          df = pd.read_csv("/mnt/data/HousingData_clean.csv")
          sns.pairplot(df)
          plt.show()
        success_message: Visualisasi selesai, tampilkan grafik.
        failure_message: Gagal membuat visualisasi, periksa data atau skrip visualisasi.

      - name: Penyimpanan Hasil
        description: |
          Menyimpan hasil analisis atau model ke file output.
        action: save_file
        output: /mnt/data/Hasil_Analisis.csv
        success_message: Hasil analisis berhasil disimpan.
        failure_message: Gagal menyimpan hasil analisis.
