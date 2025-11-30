# 🔍 BM25 + Semantic Search

Proyek ini menggabungkan **BM25** (lexical search) dengan **Semantic Search** berbasis embedding untuk menghasilkan pencarian dokumen yang lebih relevan.
Dapat digunakan untuk sistem pencarian teks, FAQ retrieval, search engine mini, dan lain-lain.

---

## 🚀 Features

✔ Hybrid Search: BM25 + Embedding Semantic Search
✔ Mudah dikembangkan dan di-custom
✔ Struktur kode sederhana
✔ Mendukung dataset teks lokal

---

## 📂 Struktur Proyek

```
BM25-Semantic-Search/
│
├── src/
│   ├── bm25_model.py
│   ├── embedding_model.py
│   ├── data_loader.py
|   ├── predict.py
│   ├── hybrid_search.py
│   └── utils.py
│
├── dataset/
│   └── data.txt  # dokumen corpus
│
├── main.py      # menjalankan demo pencarian
├── train.py     # membangun index BM25 dan embeddings
├── requirements.txt
└── README.md
```

---

## 🛠️ Cara Menjalankan Secara Manual

### 1️⃣ Clone Repository

```bash
git clone https://github.com/whochill19/BM25-Semantic-Search.git
cd BM25-Semantic-Search
```

### 2️⃣ Buat Virtual Environment (Opsional tapi direkomendasikan)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📌 Dataset

Pastikan folder `dataset/` berisi file teks:

* Setiap baris atau paragraph = 1 dokumen pencarian
* Format UTF-8 direkomendasikan

Contoh format:

```
Dokumen tentang komputer dan teknologi.
Informasi mengenai pemrograman Python.
Artikel terkait game development dan AI.
```

---

## 🧠 Build Index / Train Hybrid Search

```bash
python train.py
```

Script ini akan:

* Mengolah corpus ke token BM25
* Menghasilkan embeddings semantic search
* Menyimpan index untuk inference

---

## 🔎 Jalankan Query Search

```bash
python main.py
```

Masukkan kata kunci → hasil dokumen paling relevan akan ditampilkan

---

## 📈 Hasil Output

Contoh ketika user mengetik:

```
Query: "game artificial intelligence"

Top Results:
1. Artikel terkait game development dan AI.
2. Informasi mengenai pemrograman Python.
```

---

## 🧩 Pengembangan Selanjutnya

* Integrasi API/streamlit untuk UI search bar 🔍
* Penambahan re-ranking hasil search
* Support JSON / Database corpus
* Fine-tuning embedding model

---

## 🤝 Kontribusi

Pull request terbuka untuk siapa saja yang ingin mengembangkan proyek ini.

---

## 📜 Lisensi

MIT License © 2025
