import tkinter as tk
from tkinter import ttk
import webbrowser
import json
import os


def display_transcription_view(result, wav_path):
    words = result.get("segments", [])
    if not words:
        print("[TRANSCRIBE] Aucun mot détecté.")
        return

    json_path = os.path.splitext(wav_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[TRANSCRIBE] Résultat enregistré : {json_path}")

    root = tk.Toplevel()
    root.title("Transcription Viewer")
    root.geometry("1000x600")

    filter_threshold = tk.DoubleVar(value=0.0)

    container = tk.Text(root, wrap="word", font=("Helvetica", 11))
    container.pack(fill="both", expand=True)
    container.config(state="disabled")

    def update_display():
        threshold = filter_threshold.get()
        container.config(state="normal")
        container.delete("1.0", tk.END)
        for seg in words:
            conf = seg.get("confidence", 1.0)
            text = seg.get("text", "")
            start = seg.get("start", 0)
            if conf < threshold:
                continue
            tag = f"conf_{int(conf*100)}"
            container.insert(tk.END, f"[{start:05.2f}s] {text}\n", tag)
            container.tag_configure(tag, foreground=color_for_confidence(conf))
        container.config(state="disabled")

    def color_for_confidence(conf):
        if conf > 0.9:
            return "black"
        elif conf > 0.8:
            return "#444444"
        elif conf > 0.6:
            return "orange"
        else:
            return "red"

    slider_frame = tk.Frame(root)
    slider_frame.pack(fill="x")
    tk.Label(slider_frame, text="Seuil de confiance :").pack(side="left")
    slider = tk.Scale(slider_frame, variable=filter_threshold, from_=0.0, to=1.0,
                      resolution=0.01, orient="horizontal", length=300, command=lambda e: update_display())
    slider.pack(side="left")

    update_display()
