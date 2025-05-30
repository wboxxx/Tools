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
import tempfile
import wave

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
transcription_queue = queue.Queue()
navigation_context = ["root"]
horizontal_index = 0
last_tag_time = 0
tag_log = []
model = whisper.load_model("base")
SAMPLERATE = 16000
CHUNK_DURATION = 5
TAG_PATTERN = re.compile(r"tag\\s+(?P<type>menu|nav|act|note|capture)\\s+(?P<action>\\w+)(?:\\s+(?P<target>.+))?", re.IGNORECASE)
last_volume_print_time = 0
recorded_frames = []
recording = False


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

def save_config():
    with open(CONFIG_PATH, "w") as f:
        json.dump({"last_folder": selected_screenshot_dir}, f)

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
    global last_volume_print_time
    if status:
        Brint("[AUDIO] [WARNING]", status)
    volume = np.linalg.norm(indata) * 10
    now = time.time()
    if now - last_volume_print_time >= 1.0:
        Brint(f"[AUDIO] [VOLUME] Current level: {volume:.2f}")
        last_volume_print_time = now
    transcription_queue.put(indata.copy())

def transcription_worker():
    global last_tag_time, navigation_context, horizontal_index
    buffer = np.empty((0, 1), dtype=np.float32)
    while True:
        try:
            data = transcription_queue.get(timeout=1)
            buffer = np.append(buffer, data)
            if len(buffer) >= SAMPLERATE * CHUNK_DURATION:
                Brint("[TRANSCRIBE] [PROCESSING] Audio chunk...")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                    with wave.open(wav_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLERATE)
                        wf.writeframes((buffer[:SAMPLERATE * CHUNK_DURATION] * 32767).astype(np.int16).tobytes())
                    buffer = buffer[SAMPLERATE * CHUNK_DURATION:]
                try:
                    result = model.transcribe(wav_path, language="fr", fp16=False, temperature=0, beam_size=5)
                    os.remove(wav_path)
                    text = result.get("text", "")
                    Brint("[TRANSCRIBE] [RESULT]", text.strip())
                    for match in TAG_PATTERN.finditer(text):
                        tag_type = match.group("type").lower()
                        action = match.group("action").lower()
                        target = match.group("target") or ""
                        Brint(f"[TAG DETECTED] type={tag_type} action={action} target={target}")
                        if tag_type == "menu":
                            if action == "enter":
                                navigation_context.append(target.strip() or "Unnamed")
                            elif action == "up" and len(navigation_context) > 1:
                                navigation_context.pop()
                            elif action == "down":
                                navigation_context.append("step" + str(len(navigation_context)))
                            elif action == "left":
                                navigation_context.append("left")
                            elif action == "right":
                                navigation_context.append("right")
                        elif tag_type == "note":
                            tag_log.append({
                                "timestamp": time.time(),
                                "context": list(navigation_context),
                                "note": target.strip()
                            })
                            Brint("[NOTE] [LOGGED]", target.strip())
                except Exception as e:
                    Brint("[TRANSCRIBE] [ERROR]", str(e))
        except queue.Empty:
            continue

def launch_gui():
    def start_all():
        if not selected_screenshot_dir:
            Brint("[NAV] [ERROR] Aucun dossier sélectionné.")
            return
        start_watching_directory()
        threading.Thread(target=transcription_worker, daemon=True).start()
        try:
            stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLERATE)
            stream.start()
            Brint("[AUDIO] [STARTED] Stream actif")
        except Exception as e:
            Brint("[AUDIO] [ERROR]", str(e))

    load_config()
    root = tk.Tk()
    root.title("Live Screenshot Annotator")

    tk.Label(root, text="Sélectionne le dossier de captures d'écran :").pack(pady=10)
    tk.Button(root, text="Choisir dossier", command=select_directory).pack(pady=5)
    tk.Button(root, text="▶ Lancer la capture", command=start_all).pack(pady=20)

    root.mainloop()

launch_gui()
