import tkinter as tk
from tkinter import filedialog, messagebox
import whisper
import os

def transcribe_file():
    root = tk.Tk()
    root.withdraw()  # On ne montre pas la fenêtre principale

    file_path = filedialog.askopenfilename(
        title="Choisir un fichier .wav",
        filetypes=[("Fichiers WAV", "*.wav")]
    )

    if not file_path:
        print("Aucun fichier sélectionné.")
        return

    if not file_path.lower().endswith(".wav"):
        messagebox.showerror("Erreur", "Le fichier sélectionné n'est pas un fichier WAV.")
        return

    print(f"Transcription du fichier : {file_path}")
    model = whisper.load_model("small")  # Ou "base", "medium", "large" selon les besoins
    result = model.transcribe(file_path, language="fr")

    text_output = result["text"]
    print("\n--- Transcription ---\n")
    print(text_output)

    # Optionnel : sauvegarde dans un fichier texte
    output_path = os.path.splitext(file_path)[0] + "_transcription.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_output)
    print(f"\n✅ Transcription enregistrée dans : {output_path}")

if __name__ == "__main__":
    transcribe_file()
