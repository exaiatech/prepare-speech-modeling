# ARCHIVED - not part of the pipeline. Kept for reference only.
# Source: features.ipynb, cell 8 (0-based index in the original notebook).
# What it does: Copies English recordings by matching file stems against 'uid'.
# Why archived: Corrected version of cell 7. Not wired into the pipeline: 03_extract_features.py reads the raw download folders directly.
# The code below is a verbatim copy of the notebook cell; it was NOT refactored,
# NOT parameterised via config.yaml, and its hardcoded paths still point at the
# original machine. Do not expect it to run unchanged.

import os
import shutil
import pandas as pd
from pathlib import Path

# 1. Dosya ve klasör yollarını belirleyin
metadata_file = r"C:\Users\arday\Downloads\metadata.csv"

source_train_dir = r"C:\Users\arday\Downloads\train_audios"
source_test_dir = r"C:\Users\arday\Downloads\test_audios"

output_train_dir = r"C:\Users\arday\Desktop\aaaa\train"
output_test_dir = r"C:\Users\arday\Desktop\aaaa\test"

# 2. Çıktı klasörlerini oluşturun
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# 3. Metadata'yı okuyun ve İngilizce (english) olan 'uid' listesini alın
df = pd.read_csv(metadata_file)
# 'language' sütunundaki değerin 'english' olduğu satırları filtreliyoruz
english_data = df[df['language'] == 'english']
# Dosyaları eşleştirmek için 'uid' sütununu listeye çeviriyoruz
english_uids = set(english_data['uid'].astype(str).tolist())

# 4. Dosyaları filtreleyip kopyalayan fonksiyon
def filter_and_copy(source_dir, output_dir, valid_uids):
    if not os.path.exists(source_dir):
        print(f"Uyarı: {source_dir} klasörü bulunamadı.")
        return

    copied_count = 0
    # Kaynak klasördeki tüm dosyaları kontrol et
    for current_file in os.listdir(source_dir):
        # Dosya uzantısını ayırarak sadece kök ismini (uid) alıyoruz (örn: 'aaop.wav' -> 'aaop')
        file_stem = Path(current_file).stem
        
        # Eğer dosya ismi İngilizce UID listemizde varsa kopyala
        if file_stem in valid_uids:
            source_path = os.path.join(source_dir, current_file)
            output_path = os.path.join(output_dir, current_file)
            
            shutil.copy2(source_path, output_path)
            copied_count += 1
            
    print(f"{source_dir} klasöründen {copied_count} adet İngilizce dosya kopyalandı.")

# 5. İşlemi train ve test klasörleri için çalıştırın
print("Train dosyaları işleniyor...")
filter_and_copy(source_train_dir, output_train_dir, english_uids)

print("Test dosyaları işleniyor...")
filter_and_copy(source_test_dir, output_test_dir, english_uids)

print("İşlem tamamlandı!")
