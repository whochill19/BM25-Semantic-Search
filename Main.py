import tkinter as tk
from tkinter import ttk
import pandas as pd
import difflib

from bm25_model import BM25
from preprocess import preprocess_query
from embedding_model import load_embedding, hybrid_search


# ========================
#  LOAD DATA + INIT MODEL
# ========================
df = pd.read_csv("./dataset/processed_documents.csv")

bm25 = BM25(k1=1.5, b=0.75)
bm25.fit(df["processed_document"].tolist())

try:
    model, corpus_embeddings = load_embedding("./dataset/models")
    use_semantic = True
except Exception as e:
    print(f"Tidak bisa memuat embedding: {e}")
    model, corpus_embeddings = None, None
    use_semantic = False


# ========================
#  SPELLING CORRECTION
# ========================
def correct_spelling(query, vocab, cutoff=0.8):
    words = query.lower().split()
    corrected = []

    for w in words:
        match = difflib.get_close_matches(w, vocab, n=1, cutoff=cutoff)
        corrected.append(match[0] if match else w)

    new_query = " ".join(corrected)
    return new_query if new_query != query else None


# ========================
#  GUI APPLICATION
# ========================
class MedicineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medicine Search Engine")
        self.root.geometry("900x600")

        self.df = df

        # ---- Query Input ----
        frame = ttk.Frame(root)
        frame.pack(pady=10)

        ttk.Label(frame, text="Cari Obat / Gejala:").pack(side=tk.LEFT, padx=5)

        self.entry = ttk.Entry(frame, width=50)
        self.entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(frame, text="Search", command=self.search).pack(side=tk.LEFT, padx=5)

        # ---- Output Text ----
        self.output = tk.Text(root, wrap="word", height=25)
        self.output.pack(fill="both", expand=True, padx=10, pady=10)

        # Scrollbar
        scroll = ttk.Scrollbar(self.output, command=self.output.yview)
        self.output.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Highlight style
        self.output.tag_config("highlight", background="yellow", foreground="black")

    # ========================
    #  Highlight Helper
    # ========================
    def _insert_highlight_text(self, text, terms):
        words = text.split()

        for word in words:
            w = word.lower()
            if any(t in w for t in terms):
                self.output.insert(tk.END, word + " ", "highlight")
            else:
                self.output.insert(tk.END, word + " ")

    # ========================
    #  DISPLAY RESULT
    # ========================
    def display_results(self, results, query):
        self.output.delete("1.0", tk.END)

        self.output.insert(tk.END, f"Hasil pencarian untuk: {query}\n")
        self.output.insert(tk.END, f"{len(results)} hasil ditemukan.\n\n")

        terms = query.lower().split()

        for rank, (doc_index, score) in enumerate(results, 1):
            row = self.df.iloc[doc_index]

            name = row.get("Medicine Name", "Tidak tersedia")
            uses = str(row.get("Uses", "Tidak tersedia") or "Tidak tersedia")
            comp = str(row.get("Composition", "Tidak tersedia") or "Tidak tersedia")
            side = str(row.get("Side_effects", "Tidak tersedia") or "Tidak tersedia")

            # Judul
            self.output.insert(tk.END, f"{rank}. {name}  (score: {score:.4f})\n")

            # Uses
            self.output.insert(tk.END, "   Kegunaan     : ")
            self._insert_highlight_text(uses, terms)
            self.output.insert(tk.END, "\n")

            # Composition
            self.output.insert(tk.END, "   Komposisi    : ")
            self._insert_highlight_text(comp, terms)
            self.output.insert(tk.END, "\n")

            # Side effects
            self.output.insert(tk.END, f"   Efek samping : {side}\n\n")

    # ========================
    #  BUTTON SEARCH
    # ========================
    def search(self):
        query = self.entry.get().strip()

        if not query:
            self.output.insert(tk.END, "Masukkan kata kunci pencarian.\n")
            return

        processed = preprocess_query(query)

        # BM25 Search
        results = bm25.search(processed, top_k=10)

        if results:
            self.display_results(results, query)
            return

        # Fallback ke Semantic Embedding
        if use_semantic:
            vocab = set(" ".join(df["processed_document"]).split())
            corrected = correct_spelling(query, vocab)

            final_query = corrected if corrected else query

            results = hybrid_search(
                query=final_query,
                model=model,
                corpus_embeddings=corpus_embeddings,
                bm25=bm25,
                df=df,
                alpha=0.6,
                top_k=10
            )

            if results:
                self.display_results(results, final_query)
            else:
                self.output.insert(tk.END, "Tidak ditemukan hasil.\n")

        else:
            self.output.insert(tk.END, "BM25 tidak menemukan hasil, dan embedding tidak aktif.\n")


# ========================
#  RUN APP
# ========================
root = tk.Tk()
app = MedicineGUI(root)
root.mainloop()
