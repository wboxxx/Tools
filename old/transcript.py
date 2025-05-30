import tkinter as tk
from tkinter import filedialog, messagebox
import whisper
import os
import torch
import wave
import contextlib
from tqdm import tqdm
import time
import json
def save_processing_stats(log_path, duration_s, processing_time_s):
    with open(log_path, "w") as f:
        json.dump({
            "duration_s": duration_s,
            "processing_time_s": processing_time_s
        }, f)

def load_processing_stats(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return None

def get_audio_duration(filepath):
    with contextlib.closing(wave.open(filepath, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration
def format_timestamp(seconds: float):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"[{minutes:02}:{seconds:02}]"
from tkinter import Tk, filedialog

def select_wav_file():
    root = Tk()
    root.withdraw()  # Cache la fenêtre principale
    root.update()    # Force l'initialisation (important sous Windows)
    file_path = filedialog.askopenfilename(
        title="Choisir un fichier .wav",
        filetypes=[("Fichiers WAV", "*.wav")]
    )
    root.destroy()   # Détruit proprement le root après usage
    return file_path

def transcribe_file():
    print("Transcribe starts")

    file_path = select_wav_file()

    if not file_path:
        print("Aucun fichier sélectionné.")
        return

    if not file_path.lower().endswith(".wav"):
        messagebox.showerror("Erreur", "Le fichier sélectionné n'est pas un fichier WAV.")
        return

    print(f"📁 Fichier sélectionné : {file_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Appareil utilisé : {device.upper()}")

    model = whisper.load_model("small", device=device)
    audio_duration = get_audio_duration(file_path)
    
    log_path = "whisper_speed_log.json"
    stats = load_processing_stats(log_path)

    # Estimation temps total de traitement (si stats disponibles)
    estimated_total_time = None
    if stats:
        ratio = stats["processing_time_s"] / stats["duration_s"]
        estimated_total_time = audio_duration * ratio
        print(f"⏱️ Estimation traitement : ~{estimated_total_time:.1f}s")


    print(f"🎧 Durée de l'audio : {audio_duration:.1f} secondes")

    print("🕐 Transcription en cours...")
    progress_bar = tqdm(total=audio_duration, unit="s", desc="Progression")

    # Hack : boucle de faux avancement le temps que Whisper tourne
    # (en vrai, Whisper fait tout d’un coup)
    start_time = time.time()
    result = None
    
    loading_popup = tk.Toplevel()
    loading_popup.title("Transcription en cours")
    loading_popup.geometry("400x100")
    loading_popup.attributes("-topmost", True)
    tk.Label(loading_popup, text="⏳ Transcription en cours...", font=("Helvetica", 14)).pack(pady=20)
    loading_popup.update()

    
    try:
        result = model.transcribe(file_path, language="fr")
    finally:
        elapsed = time.time() - start_time

        # Sauvegarde stats
        save_processing_stats(log_path, audio_duration, elapsed)

        # Progression fake "réaliste"
        progress_duration = estimated_total_time or elapsed
        step_count = int(progress_duration / 0.1)
        step = audio_duration / step_count

        current = 0
        while current < audio_duration:
            time.sleep(0.1)
            current += step
            progress_bar.update(min(step, audio_duration - progress_bar.n))
        progress_bar.close()
    loading_popup.destroy()

    # Format horodaté par segment
    output_lines = []
    for segment in result["segments"]:
        ts = format_timestamp(segment["start"])
        text = segment["text"].strip()
        output_lines.append(f"{ts}\n{text}\n")

    output_text = "\n".join(output_lines)
    print(output_text)

    output_path = os.path.splitext(file_path)[0] + "_transcription.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\n✅ Transcription enregistrée dans : {output_path}")

if __name__ == "__main__":
    transcribe_file()
