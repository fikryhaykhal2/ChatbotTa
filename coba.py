import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Definisikan Path Aset
DATA_PATH = "thesis_data_features.csv"
FAISS_PATH = "data/faiss_index.bin"
SBERT_PATH = "models/sbert_model"
GENERATOR_PATH = "models/title_generator"
TOPIC_MAP_PATH = "topic_mapping.csv"

def check_file(path):
    exists = os.path.exists(path)
    print(f"[{'✅ ADA' if exists else '❌ TIDAK ADA'}] {path}")
    if not exists:
        raise FileNotFoundError(f"File atau direktori penting hilang: {path}")

try:
    print("--- Pengecekan Eksistensi File ---")
    check_file(DATA_PATH)
    check_file(FAISS_PATH)
    check_file(TOPIC_MAP_PATH)
    check_file(SBERT_PATH)
    check_file(GENERATOR_PATH)
    
    print("\n--- Pengecekan Loading Model (Debugging) ---")
    # Pengecekan loading SBERT
    sbert_model = SentenceTransformer(SBERT_PATH)
    print("✅ Model SBERT berhasil dimuat.")
    
    # Pengecekan loading FAISS Index
    index = faiss.read_index(FAISS_PATH)
    print("✅ Index FAISS berhasil dimuat.")
    
    # Pengecekan loading T5 Generator
    title_generator_pipe = pipeline("text2text-generation", 
                                   model=GENERATOR_PATH, 
                                   tokenizer=GENERATOR_PATH)
    print("✅ Model Generator T5 berhasil dimuat.")

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Struktur Direktori Gagal. Silakan buat/pindahkan file/folder yang hilang. Detail: {e}")
except Exception as e:
    print(f"\nFATAL ERROR: Gagal memuat Model/Data meskipun file ada. Ini mungkin masalah versi library atau format file. Error: {e}")