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
from tkinter import ttk
from tkinter import messagebox

transcription_display_data = []


transcription_tab_frame = None
transcription_text_widget = None

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
confidence_index = {}  # clé = mot, valeur = (tab_frame, tag)
import json
from datetime import datetime
def load_transcription_from_json():
    json_path = filedialog.askopenfilename(
        title="Charger un fichier JSON",
        filetypes=[("Fichiers JSON", "*.json")]
    )

    if not json_path:
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        transcription_text_widget.delete("1.0", tk.END)

        # Initialiser cette variable si elle n'existe pas déjà
        global transcription_display_data
        if "transcription_display_data" not in globals():
            transcription_display_data = []

        transcription_display_data.clear()

        for entry in data:
            word = entry["text"]
            start = entry["start"]
            conf = entry["confidence"]

            tag = f"word_{start:.2f}"
            transcription_display_data.append(entry)

            color = "black" if conf > 0.8 else "orange" if conf > 0.5 else "red"
            transcription_text_widget.insert(tk.END, word + " ", (tag,))
            transcription_text_widget.tag_config(tag, foreground=color)
            transcription_text_widget.tag_bind(tag, "<Button-1>", lambda e, t=start: jump_to_time(t))

        Brint(f"[LOAD] ✅ JSON chargé avec {len(data)} mots depuis : {json_path}")

    except Exception as e:
        Brint("[LOAD JSON] [ERROR]", str(e))
        messagebox.showerror("Erreur", f"Impossible de charger le fichier JSON : {str(e)}")
def save_transcription_to_json():
    if not last_transcribed_wav_path:
        messagebox.showwarning("Avertissement", "Aucune transcription à sauvegarder.")
        return

    output_path = last_transcribed_wav_path.replace(".wav", "_transcription.json")

    words_data = []
    for tag in transcription_text_widget.tag_names():
        if not tag.startswith("word_"):
            continue
        try:
            start_time = float(tag.split("_")[1])
        except ValueError:
            continue

        start_index = transcription_text_widget.tag_ranges(tag)[0]
        end_index = transcription_text_widget.tag_ranges(tag)[1]
        word_text = transcription_text_widget.get(start_index, end_index).strip()

        # Récupérer couleur = niveau de confiance
        color = transcription_text_widget.tag_cget(tag, "foreground")
        confidence = {
            "black": 1.0,
            "orange": 0.65,
            "red": 0.4
        }.get(color, 1.0)

        words_data.append({
            "text": word_text,
            "start": round(start_time, 2),
            "confidence": round(confidence, 2)
        })

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(words_data, f, indent=2, ensure_ascii=False)
        Brint(f"[SAVE] ✅ Transcription sauvegardée : {output_path}")
    except Exception as e:
        Brint("[SAVE] [ERROR]", str(e))
        messagebox.showerror("Erreur", f"Échec de la sauvegarde : {str(e)}")

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

    transcription_notebook.select(transcription_tab_frame)
    transcription_text_widget.delete("1.0", tk.END)

    # Reset index de confiance
    confidence_index.clear()
    confidence_index_list.delete(0, tk.END)

    if use_faster_var and use_faster_var.get():
        Brint("[TRANSCRIBE] [MODE] ➤ Using faster-whisper")
        try:
            import torch
            from faster_whisper import WhisperModel
            model_size = "base"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            Brint("[FASTER] Initialisation du modèle sur :", device)

            faster_model = WhisperModel(model_size, compute_type="float16", device=device)
            segments, info = faster_model.transcribe(
                wav_path, language="fr", beam_size=5, word_timestamps=True
            )

            for segment in segments:
                for word in segment.words:
                    word_text = word.word
                    start = word.start
                    conf = word.probability or 1.0

                    Brint(f"[FW] '{word_text}' @ {start:.2f}s (conf {conf:.2f})")
                    if conf < confidence_threshold.get():
                        Brint(f"[FW] Ignoré (confiance {conf:.2f})")
                        continue

                    tag = f"word_{start:.2f}"
                    color = "black" if conf > 0.8 else "orange" if conf > 0.5 else "red"

                    transcription_text_widget.insert(tk.END, word_text + " ", (tag,))
                    transcription_text_widget.tag_config(tag, foreground=color)
                    transcription_text_widget.tag_bind(tag, "<Button-1>", lambda e, t=start: jump_to_time(t))

                    if color == "red":
                        confidence_index[word_text] = tag
                        confidence_index_list.insert(tk.END, word_text)

        except Exception as e:
            Brint("[TRANSCRIBE] [ERROR - FASTER]", str(e))
            transcription_text_widget.insert(tk.END, f"[ERREUR FASTER-WHISPER] {str(e)}\n")
        return

    # Fallback Whisper classique
    Brint("[TRANSCRIBE] [MODE] ➤ Using classic Whisper")

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

                transcription_text_widget.insert(tk.END, word + " ", (tag,))
                transcription_text_widget.tag_config(tag, foreground=color)
                transcription_text_widget.tag_bind(tag, "<Button-1>", lambda e, t=start: jump_to_time(t))

                if color == "red":
                    confidence_index[word] = tag
                    confidence_index_list.insert(tk.END, word)

        for match in TAG_PATTERN.finditer(text):
            tag_type = match.group("type").lower()
            action = match.group("action").lower()
            target = match.group("target") or ""
            Brint(f"[TAG DETECTED] type={tag_type} action={action} target={target}")

    except Exception as e:
        Brint("[TRANSCRIBE] [ERROR]", str(e))
        transcription_text_widget.insert(tk.END, f"[ERREUR Whisper classique] {str(e)}\n")

def update_timer():
    global timer_running
    if timer_running:
        elapsed = int(time.time() - record_start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        timer_label.config(text=f"Durée : {minutes:02}:{seconds:02}")
        timer_label.after(1000, update_timer)


def launch_gui():
    # 🔧 Déclarations globales en premier
    global timer_label
    global audio_output_var
    global record_button
    global use_faster_var
    global confidence_threshold
    global transcription_notebook
    global confidence_index_tab
    global confidence_index_list
    global confidence_index
    global transcription_tab_frame
    global transcription_text_widget


    def toggle_output_audio():
        global CAPTURE_OUTPUT_AUDIO
        CAPTURE_OUTPUT_AUDIO = not CAPTURE_OUTPUT_AUDIO
        Brint(f"[AUDIO] [CONFIG] Capture sortie audio = {CAPTURE_OUTPUT_AUDIO}")

    def toggle_faster():
        Brint(f"[TRANSCRIBE] [OPTION] Utiliser faster-whisper = {use_faster_var.get()}")

    def update_confidence_display(val):
        Brint(f"[TRANSCRIBE] [FILTER UPDATE] Nouveau seuil de confiance = {val}")
        if last_transcribed_wav_path:
            Brint("[TRANSCRIBE] [FILTER UPDATE] Relance affichage transcription filtrée")
            transcribe_file(last_transcribed_wav_path)
        else:
            Brint("[TRANSCRIBE] [FILTER UPDATE] Aucun fichier à réafficher")

    def on_slider_release(event):
        value = confidence_threshold.get()
        Brint(f"[UI] [SLIDER RELEASED] Valeur = {value}")
        update_confidence_display(str(value))

    def reset_slider(event):
        print("[TEST] double-click detected on slider")  # DEBUG
        Brint("[UI] [SLIDER RESET] Double-clic → remise à 0.0")
        confidence_threshold.set(0.0)
        update_confidence_display("0.0")
    def start_all():
        if not selected_screenshot_dir:
            Brint("[NAV] [ERROR] Aucun dossier sélectionné.")
            return
        start_watching_directory()
        Brint("[NAV] [ACTION] ▶ Lancement du processus complet (audio + screenshot)")
        
    def on_confidence_word_click(event):
        selection = confidence_index_list.curselection()
        if not selection:
            return

        word = confidence_index_list.get(selection[0])
        tag = confidence_index.get(word)
        if tag:
            Brint(f"[INDEX] Mot sélectionné : '{word}'")
            Brint(f"[INDEX] Tag associé à '{word}' : {tag}")
            transcription_notebook.select(transcription_tab_frame)
            Brint("[INDEX] Onglet 'Transcription' sélectionné")

            ranges = transcription_text_widget.tag_nextrange(tag, "1.0")
            if ranges:
                start_index = ranges[0]
                transcription_text_widget.see(start_index)
                transcription_text_widget.tag_remove("highlight", "1.0", tk.END)
                transcription_text_widget.tag_add("highlight", start_index, ranges[1])
                transcription_text_widget.tag_config("highlight", background="yellow")
            else:
                Brint(f"[INDEX] Tag '{tag}' introuvable dans la transcription.")

        Brint(f"[INDEX] Surlignage appliqué au tag : {tag}")
    
    

    load_config()

    root = tk.Tk()
    root.title("Live Screenshot Annotator")

    # 🎛 UI : timer
    timer_label = tk.Label(root, text="Durée : 00:00")
    timer_label.pack()

    # 📂 Sélection du dossier
    tk.Label(root, text="Sélectionne le dossier de captures d'écran :").pack(pady=10)
    tk.Button(root, text="Choisir dossier", command=select_directory).pack(pady=5)

    # ▶ Démarrage annotation + audio
    tk.Button(root, text="▶ Démarrer annotation + audio", command=start_all).pack(pady=20)

    # 🎧 Sélection loopback audio
    tk.Button(root, text="🎛 Choisir sortie audio (loopback)", command=choose_loopback_device).pack(pady=5)

    # ✅ Checkbox : activer loopback
    audio_output_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="🎧 Capturer aussi le son de sortie (loopback)", variable=audio_output_var,
                   command=toggle_output_audio).pack()

    # ✅ Checkbox : activer Faster-Whisper
    use_faster_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="⚡ Utiliser Faster-Whisper (GPU optimisé)", variable=use_faster_var,
                   command=toggle_faster).pack()

    # 🔘 Bouton enregistrement
    record_button = tk.Button(root, text="⏺ Start Recording", command=toggle_record)
    record_button.pack(pady=10)

    # 🎚️ Slider de confiance
    confidence_threshold = tk.DoubleVar(value=0.0)
    tk.Label(root, text="Seuil de confiance (affichage):").pack()

    slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                      variable=confidence_threshold)
    slider.pack()
    slider.bind("<ButtonRelease-1>", on_slider_release)
    slider.bind("<Button-3>", reset_slider)
    
    # 📑 Création du notebook principal
    # 📑 Création du notebook principal
    transcription_notebook = ttk.Notebook(root)

    transcription_tab_frame = tk.Frame(transcription_notebook)
    transcription_text_widget = tk.Text(transcription_tab_frame, wrap=tk.WORD)
    transcription_text_widget.pack(fill="both", expand=True)
    # Après transcription_text_widget.pack(...)
    transcription_text_widget.pack(fill="both", expand=True)
    load_button = tk.Button(transcription_tab_frame, text="📂 Load Transcription", command=load_transcription_from_json)
    load_button.pack(pady=5, anchor="ne")

    save_button = tk.Button(transcription_tab_frame, text="💾 Save Transcription", command=save_transcription_to_json)
    save_button.pack(pady=5, anchor="ne")

    transcription_notebook.add(transcription_tab_frame, text="📝 Transcription")
    transcription_notebook.select(transcription_tab_frame)

    
    transcription_notebook.pack(fill="both", expand=True, padx=10, pady=10)

    # 🔍 Onglet Confidence Index
    confidence_index_tab = tk.Frame(transcription_notebook)
    transcription_notebook.add(confidence_index_tab, text="🔍 Confidence Index")

    confidence_index_list = tk.Listbox(confidence_index_tab)
    confidence_index_list.pack(fill="both", expand=True)
    confidence_index_list.bind("<Double-Button-1>", on_confidence_word_click)
    Brint("[UI] [BIND] Bind appliqué sur Double-Click dans confidence_index_list")


    # 🪟 Lancement de l'UI
    root.mainloop()
launch_gui()
