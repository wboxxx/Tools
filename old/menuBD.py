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

timer_label = None
record_start_time = None
timer_running = False


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
        os.startfile(os.path.dirname(wav_path))

def transcribe_file(wav_path):
    Brint("[TRANSCRIBE] [TRANSCRIPTION STARTED]", wav_path)
    try:
        result = model.transcribe(wav_path, language="fr", fp16=False, temperature=0, beam_size=5)
        text = result.get("text", "").strip()
        Brint("[TRANSCRIBE] [RESULT]", text)
        for match in TAG_PATTERN.finditer(text):
            tag_type = match.group("type").lower()
            action = match.group("action").lower()
            target = match.group("target") or ""
            Brint(f"[TAG DETECTED] type={tag_type} action={action} target={target}")
    except Exception as e:
        Brint("[TRANSCRIBE] [ERROR]", str(e))
def update_timer():
    global timer_running
    if timer_running:
        elapsed = int(time.time() - record_start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        timer_label.config(text=f"Durée : {minutes:02}:{seconds:02}")
        timer_label.after(1000, update_timer)


def launch_gui():
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


    root.mainloop()

launch_gui()
