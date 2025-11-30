import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
from bm25_model import BM25
from preprocess import preprocess_query
from embedding_model import load_embedding, hybrid_search

# --- Load Data & Models ---
df = pd.read_csv('./dataset/processed_documents.csv')

bm25 = BM25(k1=1.5, b=0.75)
bm25.fit(df['processed_document'].tolist())

try:
    model, corpus_embeddings = load_embedding('./dataset/models')
    use_semantic = True
except Exception:
    model, corpus_embeddings = None, None
    use_semantic = False

# -------------------------------------------------------------
# Autocomplete
# -------------------------------------------------------------
class AutocompleteEntry(ttk.Entry):
    def __init__(self, autocomplete_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.autocomplete_list = autocomplete_list
        self.var = self["textvariable"] = tk.StringVar()
        self.var.trace("w", self.changed)

        self.listbox = None

    def changed(self, name, index, mode):
        value = self.var.get()
        if value == "":
            if self.listbox:
                self.listbox.destroy()
            return

        matches = [w for w in self.autocomplete_list if value.lower() in w.lower()]

        if matches:
            if not self.listbox:
                self.listbox = tk.Listbox(width=40, height=5)
                self.listbox.bind("<<ListboxSelect>>", self.on_select)
                self.listbox.place(x=self.winfo_x(), y=self.winfo_y() + 28)
            self.listbox.delete(0, tk.END)
            for item in matches:
                self.listbox.insert(tk.END, item)
        else:
            if self.listbox:
                self.listbox.destroy()

    def on_select(self, event):
        if self.listbox:
            value = self.listbox.get(self.listbox.curselection())
            self.var.set(value)
            self.listbox.destroy()
            self.listbox = None

# -------------------------------------------------------------
# GUI
# -------------------------------------------------------------
class MedicineSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medicine Search Engine GUI")
        self.root.geometry("850x700")
        self.root.configure(bg="#F0F2F5")

        title = tk.Label(root, text="🔎 Medicine Search Engine", font=("Segoe UI", 18, "bold"), bg="#F0F2F5")
        title.pack(pady=10)

        autocomplete_words = list(set(df['Medicine Name'].tolist()))

        self.entry = AutocompleteEntry(autocomplete_words, root, font=("Segoe UI", 12))
        self.entry.pack(pady=10)
        self.entry.config(width=50)

        search_btn = tk.Button(root, text="Search", command=self.run_search, font=("Segoe UI", 11), bg="#4A90E2", fg="white", width=12)
        search_btn.pack(pady=5)

        self.canvas = tk.Canvas(root, bg="#F0F2F5")
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.frame = tk.Frame(self.canvas, bg="#F0F2F5")
        self.canvas.create_window((0,0), window=self.frame, anchor="nw")
        self.frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def run_search(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

        query = self.entry.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Masukkan kata kunci pencarian.")
            return

        processed_query = preprocess_query(query)
        results = bm25.search(processed_query, top_k=10)

        if not results and use_semantic:
            results = hybrid_search(
                query=query,
                model=model,
                corpus_embeddings=corpus_embeddings,
                bm25=bm25,
                df=df,
                alpha=0.6,
                top_k=10
            )

        if not results:
            tk.Label(self.frame, text="Tidak ada hasil ditemukan.", font=("Segoe UI", 12), bg="#F0F2F5").pack(pady=10)
            return

        for idx, (score, doc_id) in enumerate(results, 1):
            row = df.iloc[int(doc_id)]

            card = tk.Frame(self.frame, bg="white", bd=1, relief="solid")
            card.pack(pady=8, padx=20, fill="x")

            title = tk.Label(card, text=f"{idx}. {row['Medicine Name']}  (score: {round(score,4)})", font=("Segoe UI", 14, "bold"), bg="white", anchor="w")
            title.pack(fill="x", padx=10, pady=5)

            tk.Label(card, text=f"Kegunaan     : {row['Uses']}", bg="white", anchor="w", font=("Segoe UI", 11)).pack(fill="x", padx=10)
            tk.Label(card, text=f"Komposisi    : {row['Composition']}", bg="white", anchor="w", font=("Segoe UI", 11)).pack(fill="x", padx=10)
            tk.Label(card, text=f"Efek samping : {row['Side_effects']}", bg="white", anchor="w", font=("Segoe UI", 11)).pack(fill="x", padx=10)

            desc = row.get('description', '-')
            tk.Label(card, text=f"Deskripsi    : {desc}", bg="white", anchor="w", justify="left", wraplength=750, font=("Segoe UI", 10)).pack(fill="x", padx=10, pady=5)

if __name__ == '__main__':
    root = tk.Tk()
    app = MedicineSearchGUI(root)
    root.mainloop()