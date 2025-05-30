import tkinter as tk
from tkinter import filedialog
import os
import shutil
import threading
import queue
import time
import whisper
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import re
import json
import sounddevice as sd
import numpy as np
import wave
from faster_whisper import WhisperModel  # Ajoute en haut du fichier

transcription_display = None

SELECTED_LOOPBACK_DEVICE_INDEX = None

# Debug flags for Brint
DEBUG_FLAGS = {
    "BRINT": True,
    "TRANSCRIBE": True,
    "SCREENSHOT": True,
    "NAV": True,
    "NOTE": True,
    "AUDIO": True
}

CONFIG_PATH = "config.json"
selected_screenshot_dir = ""
output_dir = "output"
navigation_context = ["root"]
horizontal_index = 0
tag_log = []
model = whisper.load_model("base")
SAMPLERATE = 16000
TAG_PATTERN = re.compile(r"tag\\s+(?P<type>menu|nav|act|note|capture)\\s+(?P<action>\\w+)(?:\\s+(?P<target>.+))?", re.IGNORECASE)
recording = False
recorded_frames = []
stream = None
audio_output_var = None  # Variable liée à la checkbox loopback
CAPTURE_OUTPUT_AUDIO = False
use_faster_var = None

timer_label = None
record_start_time = None
timer_running = False
confidence_threshold = None
last_transcribed_wav_path = None


def Brint(*args, **kwargs):
    if not args:
        return
    first_arg = str(args[0])
    tags = re.findall(r"\[(.*?)\]", first_arg)
    if DEBUG_FLAGS.get("BRINT", None) is False:
        return
    if DEBUG_FLAGS.get("BRINT", None) is True:
        print(*args, **kwargs)
        return
    if not tags:
        print(*args, **kwargs)
        return
    for tag_str in tags:
        keywords = tag_str.upper().split()
        if any(DEBUG_FLAGS.get(kw, False) for kw in keywords):
            print(*args, **kwargs)
            return
# 🔁 Recalcul uniquement quand l'utilisateur relâche la souris
def on_slider_release(event):
    value = confidence_threshold.get()
    Brint(f"[UI] [SLIDER RELEASED] Valeur = {value}")
    update_confidence_display(str(value))




def jump_to_time(timestamp_seconds):
    Brint(f"[NAV] [JUMP TO] Clic sur mot → {timestamp_seconds:.2f}s")
    # À l’avenir : intégration player / audio seek ici

def choose_loopback_device():
    devices = sd.query_devices()
    output_devices = [d for d in devices if d['hostapi'] == sd.default.hostapi and d['max_output_channels'] > 0]
    
    selection_window = tk.Toplevel()
    selection_window.title("Choisir un périphérique de sortie (loopback)")
    selection_window.geometry("500x400")
    
    tk.Label(selection_window, text="Sélectionne le périphérique pour capturer le son de sortie :").pack(pady=10)
    
    listbox = tk.Listbox(selection_window, width=80)
    listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    for i, d in enumerate(output_devices):
        listbox.insert(tk.END, f"{i}: {d['name']}")

    def on_select():
        idx = listbox.curselection()
        if not idx:
            return
        index = int(idx[0])
        device_info = output_devices[index]
        global SELECTED_LOOPBACK_DEVICE_INDEX
        SELECTED_LOOPBACK_DEVICE_INDEX = device_info['index']
        Brint(f"[AUDIO] [SELECTED DEVICE] {device_info['name']} @ index {SELECTED_LOOPBACK_DEVICE_INDEX}")
        selection_window.destroy()

    tk.Button(selection_window, text="OK", command=on_select).pack(pady=5)

def save_config():
    with open(CONFIG_PATH, "w") as f:
        json.dump({"last_folder": selected_screenshot_dir}, f)
    Brint("[NAV] [CONFIG SAVED]", selected_screenshot_dir)


def load_config():
    global selected_screenshot_dir
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
            selected_screenshot_dir = data.get("last_folder", "")
            Brint("[NAV] [CONFIG LOADED]", selected_screenshot_dir)


def select_directory():
    global selected_screenshot_dir
    selected_screenshot_dir = filedialog.askdirectory()
    Brint("[NAV] [DIR SELECTED] Watching folder:", selected_screenshot_dir)
    save_config()


class ScreenshotHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return
        Brint("[SCREENSHOT] [NEW] Detected:", event.src_path)
        handle_new_screenshot(event.src_path)


def start_watching_directory():
    event_handler = ScreenshotHandler()
    observer = Observer()
    observer.schedule(event_handler, selected_screenshot_dir, recursive=False)
    observer.start()
    Brint("[SCREENSHOT] [WATCHING] Folder observer started")


def handle_new_screenshot(image_path):
    global navigation_context
    path = os.path.join(output_dir, *navigation_context)
    os.makedirs(path, exist_ok=True)
    dest = os.path.join(path, os.path.basename(image_path))
    shutil.copy(image_path, dest)
    Brint("[SCREENSHOT] [COPIED]", image_path, "→", dest)
    update_json_structure(dest)


def update_json_structure(image_path):
    structure_path = os.path.join(output_dir, "structure.json")
    try:
        with open(structure_path, "r") as f:
            structure = json.load(f)
    except:
        structure = {}
    ptr = structure
    for level in navigation_context:
        ptr = ptr.setdefault(level, {})
    ptr.setdefault("screenshots", []).append(os.path.basename(image_path))
    with open(structure_path, "w") as f:
        json.dump(structure, f, indent=2)
    Brint("[NAV] [JSON UPDATED] Added", image_path)


def audio_callback(indata, frames, time_info, status):
    global recorded_frames
    if status:
        Brint("[AUDIO] [WARNING]", status)
    recorded_frames.append(indata.copy())


def toggle_record():
    global recording, stream, recorded_frames
    if not recording:
        recorded_frames = []
        global record_start_time, timer_running
        record_start_time = time.time()
        timer_running = True
        update_timer()

        global CAPTURE_OUTPUT_AUDIO
        loopback = CAPTURE_OUTPUT_AUDIO
        Brint("[AUDIO] [SETUP] loopback =", loopback)

        try:
            if loopback:
                Brint("[AUDIO] [SETUP] loopback = True")
                device_index = SELECTED_LOOPBACK_DEVICE_INDEX
                if device_index is None:
                    Brint("[AUDIO] [ERROR] Aucun périphérique sélectionné pour loopback.")
                    return
                stream = sd.InputStream(
                    samplerate=SAMPLERATE,
                    channels=1,
                    dtype='float32',
                    callback=audio_callback,
                    device=device_index,
                    loopback=True
                )
            else:
                Brint("[AUDIO] [SETUP] loopback = False")
                stream = sd.InputStream(
                    callback=audio_callback,
                    channels=1,
                    samplerate=SAMPLERATE
                )

        except Exception as e:
            Brint("[AUDIO] [ERROR] Échec de la configuration du stream :", str(e))
            return

        stream.start()
        recording = True
        Brint("[AUDIO] [RECORDING] ➤ Démarrage enregistrement micro")
        record_button.config(text="⏹ Stop + Transcribe")
    else:
        stream.stop()
        stream.close()
        recording = False
        timer_running = False

        record_button.config(text="⏺ Start Recording")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wav_path = os.path.join(os.path.expanduser("~"), "Videos", f"capture_{timestamp}.wav")
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLERATE)
            wf.writeframes((np.concatenate(recorded_frames) * 32767).astype(np.int16).tobytes())
        Brint("[AUDIO] [RECORDING] ✅ Enregistrement terminé :", wav_path)
        transcribe_file(wav_path)
        # os.startfile(os.path.dirname(wav_path))
def transcribe_file(wav_path):
    global last_transcribed_wav_path
    last_transcribed_wav_path = wav_path
    Brint("[TRANSCRIBE] [TRANSCRIPTION STARTED]", wav_path)

    if use_faster_var and use_faster_var.get():
        Brint("[TRANSCRIBE] [MODE] ➤ Using faster-whisper")
        try:
            from faster_whisper import WhisperModel
            model_size = "base"  # tu peux choisir ici : "tiny", "small", "medium", "large-v2"...

            device = "cuda" if torch.cuda.is_available() else "cpu"
            Brint("[FASTER] Initialisation du modèle sur :", device)
            faster_model = WhisperModel(model_size, compute_type="float16", device=device)

            segments, info = faster_model.transcribe(
                wav_path, language="fr", beam_size=5, word_timestamps=True
            )

            if transcription_display:
                transcription_display.delete(1.0, tk.END)
                for segment in segments:
                    for word in segment.words:
                        text = word.word
                        start = word.start
                        conf = word.probability or 1.0

                        Brint(f"[FW] '{text}' @ {start:.2f}s (conf {conf:.2f})")
                        if conf < confidence_threshold.get():
                            Brint(f"[FW] Ignoré (confiance {conf:.2f})")
                            continue

                        tag = f"word_{start:.2f}"
                        color = "black" if conf > 0.8 else "orange" if conf > 0.5 else "red"

                        transcription_display.insert(tk.END, text + " ", (tag,))
                        transcription_display.tag_config(tag, foreground=color)
                        transcription_display.tag_bind(tag, "<Button-1>", lambda e, t=start: jump_to_time(t))

        except Exception as e:
            Brint("[TRANSCRIBE] [ERROR - FASTER]", str(e))
            if transcription_display:
                transcription_display.insert(tk.END, f"[ERREUR FASTER-WHISPER] {str(e)}\n")
        return

    # Fallback Whisper classique
    Brint("[TRANSCRIBE] [MODE] ➤ Using classic Whisper")

    # Debug CUDA + Triton (non bloquant)
    try:
        import torch
        Brint("[CUDA] torch.cuda.is_available():", torch.cuda.is_available())
        Brint("[CUDA] torch.version.cuda:", torch.version.cuda)
        Brint("[CUDA] torch.cuda.get_device_name():", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

        import triton
        Brint("[TRITON] version:", triton.__version__)
    except Exception as e:
        Brint("[TRANSCRIBE] [DEBUG WARNING] Pas de Triton ou CUDA:", str(e))

    try:
        result = model.transcribe(
            wav_path,
            language="fr",
            fp16=False,
            temperature=0,
            beam_size=5,
            word_timestamps=True
        )

        text = result.get("text", "").strip()
        Brint("[TRANSCRIBE] [TEXT RESULT]", text)

        words = result.get("segments", [])
        if transcription_display:
            Brint("[TRANSCRIBE] [DISPLAY] Effacement zone texte")
            transcription_display.delete(1.0, tk.END)

            for seg in words:
                for word_info in seg["words"]:
                    word = word_info["word"]
                    start = word_info["start"]
                    conf = word_info.get("probability", 1.0)
                    Brint(f"[TRANSCRIBE] [WORD] '{word}' @ {start:.2f}s (conf: {conf:.2f})")

                    if conf < confidence_threshold.get():
                        Brint(f"[TRANSCRIBE] [FILTERED OUT] '{word}' ignoré (conf {conf:.2f})")
                        continue

                    tag = f"word_{start:.2f}"
                    color = "black" if conf > 0.8 else "orange" if conf > 0.5 else "red"

                    transcription_display.insert(tk.END, word + " ", (tag,))
                    transcription_display.tag_config(tag, foreground=color)
                    transcription_display.tag_bind(tag, "<Button-1>", lambda e, t=start: jump_to_time(t))

        for match in TAG_PATTERN.finditer(text):
            tag_type = match.group("type").lower()
            action = match.group("action").lower()
            target = match.group("target") or ""
            Brint(f"[TAG DETECTED] type={tag_type} action={action} target={target}")

    except Exception as e:
        Brint("[TRANSCRIBE] [ERROR]", str(e))
        if transcription_display:
            transcription_display.insert(tk.END, f"[ERREUR Whisper classique] {str(e)}\n")

def update_timer():
    global timer_running
    if timer_running:
        elapsed = int(time.time() - record_start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        timer_label.config(text=f"Durée : {minutes:02}:{seconds:02}")
        timer_label.after(1000, update_timer)


def launch_gui():
    
 

    def toggle_faster():
        Brint(f"[TRANSCRIBE] [OPTION] Utiliser faster-whisper = {use_faster_var.get()}")

    tk.Checkbutton(root, text="⚡ Utiliser Faster-Whisper (GPU optimisé)", variable=use_faster_var,
                   command=toggle_faster).pack()

    
    
    
    def toggle_output_audio():
        global CAPTURE_OUTPUT_AUDIO
        CAPTURE_OUTPUT_AUDIO = not CAPTURE_OUTPUT_AUDIO
        Brint(f"[AUDIO] [CONFIG] Capture sortie audio = {CAPTURE_OUTPUT_AUDIO}")

    
    
    def start_all():
        
     
        if not selected_screenshot_dir:
            Brint("[NAV] [ERROR] Aucun dossier sélectionné.")
            return
        start_watching_directory()
        Brint("[NAV] [ACTION] ▶ Lancement du processus complet (audio + screenshot)")

    load_config()
    root = tk.Tk()
    

    global timer_label
    timer_label = tk.Label(root, text="Durée : 00:00")
    timer_label.pack()

   # 💡 Ta case à cocher pour faster-whisper doit venir ici, après root = tk.Tk()
    global use_faster_var
    use_faster_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="⚡ Utiliser Faster-Whisper (GPU optimisé)",
                   variable=use_faster_var, command=toggle_faster).pack()

    root.title("Live Screenshot Annotator")

    tk.Label(root, text="Sélectionne le dossier de captures d'écran :").pack(pady=10)
    tk.Button(root, text="Choisir dossier", command=select_directory).pack(pady=5)
    tk.Button(root, text="▶ Démarrer annotation + audio", command=start_all).pack(pady=20)
    tk.Button(root, text="🎛 Choisir sortie audio (loopback)", command=choose_loopback_device).pack(pady=5)

    global audio_output_var
    audio_output_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="🎧 Capturer aussi le son de sortie (loopback)", command=toggle_output_audio).pack()

    global record_button
    record_button = tk.Button(root, text="⏺ Start Recording", command=toggle_record)
    record_button.pack(pady=10)
    global transcription_display
    transcription_display = tk.Text(root, wrap=tk.WORD, height=15, width=80)
    transcription_display.pack(pady=10)
    transcription_display.insert(tk.END, "[TRANSCRIPTION] En attente d’un enregistrement...\n")
    global confidence_threshold
    confidence_threshold = tk.DoubleVar(value=0.0)

    def update_confidence_display(val):
        Brint(f"[TRANSCRIBE] [FILTER UPDATE] Nouveau seuil de confiance = {val}")
        if last_transcribed_wav_path:
            Brint("[TRANSCRIBE] [FILTER UPDATE] Relance affichage transcription filtrée")
            transcribe_file(last_transcribed_wav_path)
        else:
            Brint("[TRANSCRIBE] [FILTER UPDATE] Aucun fichier à réafficher")

    tk.Label(root, text="Seuil de confiance (affichage):").pack()
    # tk.Scale(root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
             # variable=confidence_threshold, command=on_slider_release).pack()
    slider = tk.Scale(
        root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
        variable=confidence_threshold
    )
    slider.pack()

    # 🔁 Double-clic gauche pour reset
    def reset_slider(event):
        Brint("[UI] [SLIDER RESET] Double-clic → remise à 0.0")
        confidence_threshold.set(0.0)
        update_confidence_display("0.0")

    slider.bind("<Double-Button-1>", reset_slider)
    slider.bind("<ButtonRelease-1>", on_slider_release)



    root.mainloop()

launch_gui()
