import os
import pandas as pd

def load_and_preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Fungsi untuk:
    1. Membaca dataset mentah dari input_path.
    2. Melakukan preprocessing (mengisi missing value dengan median).
    3. Menyimpan dataset yang sudah bersih ke output_path.
    4. Mengembalikan DataFrame yang sudah dipreprocess.
    """
    # 1. Load data mentah
    df = pd.read_csv(input_path)

    # 2. Preprocessing: isi missing value dengan median tiap kolom numerik
    df_clean = df.fillna(df.median(numeric_only=True))

    # 3. Simpan hasil ke output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    return df_clean


if __name__ == "__main__":
    # Gunakan path dengan forward slash (/) supaya jalan di Linux (GitHub Actions)
    input_path = "namadataset_raw/HousingData.csv"
    output_path = "preprocessing/HousingData_clean.csv"

    df_clean = load_and_preprocess(input_path, output_path)

    print("Preprocessing otomatis selesai.")
    print(f"Dataset bersih disimpan di: {output_path}")
    print("Shape dataset bersih:", df_clean.shape)
    print("5 baris pertama dataset bersih:")
    print(df_clean.head())