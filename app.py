import streamlit as st
import pandas as pd
import faiss
import numpy as np
import os
import gdown 
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import zipfile # Diperlukan untuk mengekstrak file ZIP
# import tarfile # Gunakan ini jika Anda menggunakan .tar.gz

# --- KONFIGURASI ID GOOGLE DRIVE (Telah Diperbaiki) ---
# Menggunakan ID file MURNI, bukan URL lengkap.
GDRIVE_IDS = {
    "FAISS_INDEX": "1yAExAsynzmDAIPGfnk0yv96Gb1xkgcBr", 
    "SBERT_MODEL_ZIP": "1ZsEe4FnaQeE88NlzWj8K9sgMKI6zsKlz", 
    "GENERATOR_MODEL_ZIP": "1jsZgd29BBxxV_I9FW3S6EmKsVtldtJfP", 
    "DATA_FEATURES": "1Gi8lNdDy0zTsxdnRiMrdvTnIJwHnVm2A", 
    "TOPIC_MAP": "1CVLwYRkocqT5O_31SW1LhDFI1tnBrgAl",
}
# -----------------------------------

# --- FUNGSI BARU UNTUK MENGUNDUH ASET ---
def download_assets():
    """Mengunduh model dan index dari Google Drive menggunakan gdown."""
    st.info("Mengunduh model dan index dari Google Drive...")
    
    # 1. Pastikan direktori ada
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    try:
        # Mengunduh Data dan Mapping (file kecil)
        if not os.path.exists("thesis_data_features.csv"):
             gdown.download(id=GDRIVE_IDS["DATA_FEATURES"], output="thesis_data_features.csv", quiet=True)
        if not os.path.exists("topic_mapping.csv"):
             gdown.download(id=GDRIVE_IDS["TOPIC_MAP"], output="topic_mapping.csv", quiet=True)
        
        # Mengunduh Index FAISS
        if not os.path.exists("data/faiss_index.bin"):
            gdown.download(id=GDRIVE_IDS["FAISS_INDEX"], output="data/faiss_index.bin", quiet=True)

        # 2. Mengunduh dan Mengekstrak Model SBERT
        if not os.path.isdir("models/sbert_model"):
            sbert_zip_path = "models/sbert_model.zip"
            gdown.download(id=GDRIVE_IDS["SBERT_MODEL_ZIP"], output=sbert_zip_path, quiet=True)
            
            # Ekstraksi menggunakan zipfile
            with zipfile.ZipFile(sbert_zip_path, 'r') as zip_ref:
                zip_ref.extractall("models/")
            os.remove(sbert_zip_path) # Hapus file zip setelah diekstrak

        # 3. Mengunduh dan Mengekstrak Model GENERATOR T5
        if not os.path.isdir("models/title_generator"):
            generator_zip_path = "models/generator.zip"
            gdown.download(id=GDRIVE_IDS["GENERATOR_MODEL_ZIP"], output=generator_zip_path, quiet=True)
            
            with zipfile.ZipFile(generator_zip_path, 'r') as zip_ref:
                zip_ref.extractall("models/")
            os.remove(generator_zip_path) # Hapus file zip setelah diekstrak


        st.success("Pengunduhan dan Ekstraksi Aset Selesai.")
        
    except Exception as e:
        # Error yang ditangkap di sini akan ditampilkan di Streamlit Cloud
        st.error(f"Gagal mengunduh/mengekstrak aset. Pastikan izin Google Drive diatur ke 'Siapa saja yang memiliki link'. Error: {e}")
        raise e 


# --- 1. Konfigurasi dan Memuat Aset ---
@st.cache_resource
def load_assets_into_state():
    
    # JALANKAN FUNGSI DOWNLOAD TERLEBIH DAHULU
    download_assets() # Ini memastikan semua file ada di path lokal
    
    # Path Aset
    DATA_PATH = "thesis_data_features.csv"
    FAISS_PATH = "data/faiss_index.bin"
    SBERT_PATH = "models/sbert_model"
    GENERATOR_PATH = "models/title_generator"
    TOPIC_MAP_PATH = "topic_mapping.csv"
    
    # Pemeriksaan path (Sekarang harus lulus jika download_assets berhasil)
    if not os.path.exists(DATA_PATH): raise FileNotFoundError(f"File data tidak ditemukan: {DATA_PATH}")
    if not os.path.exists(FAISS_PATH): raise FileNotFoundError(f"Index FAISS tidak ditemukan: {FAISS_PATH}")
    if not os.path.isdir(SBERT_PATH): raise FileNotFoundError(f"Direktori SBERT tidak ditemukan: {SBERT_PATH}")
    if not os.path.isdir(GENERATOR_PATH): raise FileNotFoundError(f"Direktori Generator T5 tidak ditemukan: {GENERATOR_PATH}")
        
    try:
        # Muat Aset
        data_df = pd.read_csv(DATA_PATH)
        topic_labels_df = pd.read_csv(TOPIC_MAP_PATH)
        topic_labels_map = topic_labels_df.set_index('topic_label_id')['topic_name'].to_dict()

        sbert_model_loaded = SentenceTransformer(SBERT_PATH) 
        faiss_index = faiss.read_index(FAISS_PATH)
        
        title_generator_pipe_loaded = pipeline("text2text-generation", 
                                       model=GENERATOR_PATH, 
                                       tokenizer=GENERATOR_PATH,
                                       trust_remote_code=True)
        
        # Simpan SEMUA aset ke st.session_state
        st.session_state['data'] = data_df
        st.session_state['index'] = faiss_index
        st.session_state['sbert_model'] = sbert_model_loaded
        st.session_state['title_generator_pipe'] = title_generator_pipe_loaded
        st.session_state['topic_labels'] = topic_labels_map
        st.session_state['assets_loaded'] = True
        
    except Exception as e:
        st.session_state['assets_loaded'] = False
        st.session_state['error_detail'] = str(e)


# --- INITIAL SETUP & ERROR HANDLING ---
st.set_page_config(page_title="IT Thesis Helper", layout="wide")

# Jalankan loader
if 'assets_loaded' not in st.session_state:
    load_assets_into_state()
    
# Tampilkan Error jika gagal
if not st.session_state.get('assets_loaded', False):
    st.title("❌ GAGAL MEMUAT ASET UTAMA")
    st.error(f"Aplikasi berhenti karena error saat memuat model/data. Detail: {st.session_state.get('error_detail', 'Tidak ada detail error yang tersedia.')}")
    st.stop() # Hentikan eksekusi jika gagal

# --- 2. Fungsi Logika Inti (Mengakses st.session_state secara langsung) ---

# Akses aset yang disimpan di state
data = st.session_state['data']
sbert_model = st.session_state['sbert_model']
index = st.session_state['index']
title_generator_pipe = st.session_state['title_generator_pipe']
topic_labels = st.session_state['topic_labels']


def retrieve_theses(query_text, k=5):
    """Mencari skripsi terkait menggunakan SBERT + FAISS."""
    query_vector = sbert_model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vector, k)
    results = data.iloc[indices[0]].copy()
    results['similarity_score'] = 1 - distances[0]
    return results

def generate_title(abstract_text, max_length=30):
    """Menghasilkan judul menggunakan model Seq2Seq."""
    prompt = f"generate title: {abstract_text}"
    generated = title_generator_pipe(prompt, 
                                      max_length=max_length, 
                                      num_beams=5, 
                                      do_sample=True, 
                                      top_k=50)
    return generated[0]['generated_text'].strip()

def get_top_topics(data, topic_labels):
    """Menghitung dan menampilkan topik paling populer dari dataset."""
    topic_counts = data['topic_label_id'].value_counts().head(5)
    top_topics = [f"{topic_labels.get(id, 'N/A')} ({count} Skripsi)" 
                  for id, count in topic_counts.items()]
    return top_topics


# --- 3. Antarmuka Pengguna Streamlit (UI) ---

st.title("🎓 Prototipe Chatbot Pembantu Skripsi TI")
st.caption("Sistem berbasis NLP untuk Saran Topik, Metode, dan Judul.")

st.sidebar.title("Pilih Modul")
feature = st.sidebar.radio("Navigasi:", [
    "FAQ & Topik Populer", 
    "Saran Referensi Skripsi (Retrieval)", 
    "Generator Judul Tugas Akhir"
])

st.sidebar.markdown("---")
st.sidebar.warning("Sistem ini adalah prototipe akademis. Selalu konsultasikan hasil dengan Dosen Pembimbing.")


# --- TAMPILAN FITUR 1: FAQ & Topik Populer ---
if feature == "FAQ & Topik Populer":
    st.header("📚 Analisis Topik & Prosedur")
    
    st.subheader("Tren Topik Skripsi Populer")
    top_topics = get_top_topics(data, topic_labels) 
    st.markdown("- " + "\n- ".join(top_topics))
    
    st.markdown("---")
    
    st.subheader("FAQ Prosedur & Metode Populer")
    faq_query = st.selectbox("Pilih Pertanyaan:", [
        "Apa syarat minimal untuk mengajukan seminar proposal?", 
        "Berikan contoh metode yang sering dipakai untuk 'Klasifikasi Citra'.",
        "Apa perbedaan RNN dan LSTM?"
    ])
    
    if faq_query:
        if "syarat minimal" in faq_query:
            st.info("Jawaban: Syarat meliputi minimal IPK 3.0, telah menempuh 120 SKS, dan lulus mata kuliah Metodologi Penelitian.")
        elif "metode yang sering dipakai" in faq_query:
            classification_topic_id = 0 
            popular_methods = data[data['topic_label_id']==classification_topic_id]['methods_extracted'].str.upper().tolist()
            st.info(f"Jawaban: Beberapa metode populer untuk topik tersebut meliputi: {', '.join(np.unique(popular_methods)[:5])}")
        else:
            st.info("Jawaban: **RNN** (Recurrent Neural Network) adalah jaringan dasar untuk data sekuensial, sedangkan **LSTM** (Long Short-Term Memory) adalah varian RNN yang mampu mengatasi masalah *vanishing gradient* dan mengingat informasi jangka panjang.")

            
# --- TAMPILAN FITUR 2: Saran Referensi Skripsi (Retrieval) ---
elif feature == "Saran Referensi Skripsi (Retrieval)":
    st.header("🔍 Saran Skripsi Serupa (Retrieval Module)")
    st.markdown("Masukkan deskripsi atau abstrak singkat untuk mencari referensi yang relevan.")
    
    query = st.text_area("Deskripsikan ide/topik skripsi Anda:", 
                        placeholder="Contoh: Implementasi Deep Learning untuk deteksi penyakit retina pada citra medis...")
    
    k_result = st.slider("Jumlah Skripsi yang Direkomendasikan (k):", 1, 10, 5)
    
    if st.button("Cari Skripsi Serupa"):
        if query:
            with st.spinner('Mencari referensi terdekat dengan SBERT dan FAISS...'):
                results = retrieve_theses(query, k=k_result)
                
            st.success(f"Ditemukan {len(results)} skripsi terkait:")
            
            for _, row in results.iterrows():
                with st.expander(f"**{row['judul']}** (Tahun: {row['tahun']})"):
                    st.markdown(f"**Similarity Score:** *{row['similarity_score']:.4f}*") 
                    st.markdown(f"**Metode Ekstraksi:** `{row['methods_extracted'].upper()}`")
                    st.markdown(f"**Topik Klasifikasi:** *{topic_labels.get(row['topic_label_id'])}*")
                    st.markdown(f"**Abstrak (Potongan):** {row['abstrak'][:300]}...") 

# --- TAMPILAN FITUR 3: Generator Judul Tugas Akhir ---
elif feature == "Generator Judul Tugas Akhir":
    st.header("💡 Generator Judul Otomatis (Generation Module)")
    
    input_abstract = st.text_area("Masukkan deskripsi singkat/abstrak rencana skripsi Anda:", 
                                placeholder="Contoh: Penelitian ini bertujuan untuk membandingkan performa antara metode CNN ResNet50 dan VGG16 dalam tugas klasifikasi citra satelit untuk deteksi kebakaran hutan di Kalimantan...")
    
    if st.button("Generate Judul"):
        if input_abstract:
            with st.spinner('Menghasilkan judul menggunakan model T5...'):
                generated_title = generate_title(input_abstract)
                
            st.subheader("Saran Judul Tugas Akhir (Generated):")
            st.success(generated_title.title()) 
            
            st.markdown("---")
            st.info("Model generation dilatih dari abstrak ke judul. Judul yang dihasilkan adalah saran yang perlu disempurnakan.")