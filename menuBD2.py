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
from chunk_transcriber import transcribe_wav_in_chunks
from tkinter import ttk
from tkinter import messagebox
import json
from datetime import datetime
tagged_text_widget = None
tagged_tab_frame = None
from menu_tree_builder import *
from menu_html_utils import *
from utils import Brint  # ou adapte selon ton import
from PIL import Image, ImageDraw
import webbrowser
from typing import List
import openai





# Ajoute ceci au début de ton script ou dans une section de config
import os
from tkinter import simpledialog

# Fonction pour charger ou demander la clé OpenAI

# Ajoute ceci au début de ton script ou dans une section de config
import os
import openai
from huggingface_hub import login
import os
print(os.getcwd())

def load_huggingface_token():
    try:
        token_path = os.path.join(os.path.dirname(__file__), ".hf_token")
        with open(token_path, "r", encoding="utf-8") as f:
            token = f.read().strip()
            if token.startswith("hf_"):
                login(token)
                print("[HF] ✅ Token HuggingFace chargé avec succès")
                return True
    except Exception as e:
        print("[HF] ❌ Erreur lecture token :", e)
    return False

load_huggingface_token()

def load_openai_api_key_from_file():
    """
    Charge la clé OpenAI depuis le fichier .openai_key si présent.
    Associe la clé à openai.api_key et à la variable d'environnement.
    Retourne True si la clé a été chargée, sinon False.
    """
    try:
        token_path = os.path.join(os.path.dirname(__file__), ".openai_key")
        with open(token_path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                    os.environ["OPENAI_API_KEY"] = key
                    openai.api_key = key
                    print("[OPENAI] Clé API chargée avec succès depuis .openai_key")
                    return True
    except Exception as e:
        print(f"[OPENAI] Erreur lecture clé API: {e}")
    print("[OPENAI] Clé non présente ou vide")
    return False

# Appelle cette fonction tôt dans ton script pour initialiser la clé OpenAI
load_openai_api_key_from_file()
import os
import openai
from tkinter import simpledialog

def load_openai_api_key_from_file():
    try:
        if os.path.exists(".openai_key"):
            with open(".openai_key", "r") as f:
                key = f.read().strip()
                if key:
                    os.environ["OPENAI_API_KEY"] = key
                    openai.api_key = key
                    return True
    except Exception as e:
        print(f"[OPENAI] Erreur lecture clé API: {e}")
    return False


# Appelle cette fonction très tôt dans ton code (après les imports par exemple)
# load_openai_api_key()
load_openai_api_key_from_file()




# 1. visit hf.co/pyannote/speaker-diarization and accept user conditions
# 2. visit hf.co/pyannote/segmentation and accept user conditions
# 3. visit hf.co/settings/tokens to create an access token
# 4. instantiate pretrained speaker diarization pipeline
from pyannote.audio import Pipeline
# use_auth_token="secrettoken")


# apply the pipeline to an audio file
# diarization = pipeline("audio.wav")

# dump the diarization output to disk using RTTM format
# with open("audio.rttm", "w") as rttm:
    # diarization.write_rttm(rttm)



last_loaded_session = None


action_keywords = {
    "menu": "blue",
    "root": "green",
    "down": "green",
    "side": "green",
    "up": "green",
    "tango": "orange",
    "click": "blue",
    "swipe": "blue",
    "open": "blue"
    # … autres actions
}

target_keywords = {
    "profil": "purple",
    "paramètres": "purple",
    "aide": "purple",
    "boutique": "purple"
    # … autres cibles
}
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
config_data = {"screenshot_dir": ""}
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
diarization_var = None
send_to_openai_var = None

timer_label = None
record_start_time = None
timer_running = False
confidence_threshold = None
last_transcribed_wav_path = None
confidence_index = {}  # clé = mot, valeur = (tab_frame, tag)

def format_tags_for_display(raw_text: str) -> List[int]:
    Brint(f"[TAG FORMATTER] Raw text length: {len(raw_text)}")
    pattern = re.compile(r'(?<!^)(?<!\r\n)(?<!\n)(?<!\r)(\[TANGO\])') # Pattern remains as is
    indices = []
    for match in pattern.finditer(raw_text):
        indices.append(match.start(0)) # MODIFIED LINE: Get start index of the entire match for '[TANGO]'
    Brint(f"[TAG FORMATTER] Found indices for TANGO: {indices}")
    indices.sort(reverse=True) # Sort in reverse order for safe insertion
    return indices





def move_console_to_right_half():
    import ctypes

    user32 = ctypes.windll.user32
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
        width = screen_width // 2
        height = screen_height
        x = screen_width - width
        y = 0
        ctypes.windll.user32.MoveWindow(hwnd, x, y, width, height, True)




def get_screenshots_with_timestamps():
    import os
    import re

    Brint(f"[SCREENSHOT] Scanning {selected_screenshot_dir}")
    screenshots = []
    pattern = re.compile(r"(\d+)_(\d+)_(\d+)")  # ex: 00_01_23

    if not selected_screenshot_dir:
        return []

    for fname in os.listdir(selected_screenshot_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            match = pattern.search(fname)
            if match:
                h, m, s = map(int, match.groups())
                seconds = h * 3600 + m * 60 + s
                screenshots.append((seconds, fname))
    screenshots_sorted = sorted(screenshots)
    Brint(f"[SCREENSHOT] Found {len(screenshots_sorted)} screenshots")
    return screenshots_sorted


def select_directory():
    global selected_screenshot_dir
    selected_screenshot_dir = filedialog.askdirectory()
    if selected_screenshot_dir:
        Brint("[NAV] [DIR SELECTED] Watching folder:", selected_screenshot_dir)
        save_config()

def load_config():
    global selected_screenshot_dir
    try:
        if not os.path.exists(CONFIG_PATH):
            Brint("[CONFIG] Aucun fichier de config trouvé. Valeurs par défaut utilisées.")
            return

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            # Support legacy key or new key
            selected_screenshot_dir = config.get("screenshot_dir") or config.get("last_folder", "")
            config_data["screenshot_dir"] = selected_screenshot_dir
            Brint(f"[CONFIG] 📂 Dossier images chargé : {selected_screenshot_dir}")
    except Exception as e:
        Brint("[CONFIG] ❌ Erreur de chargement :", str(e))


def update_folder_path_label():
    if selected_screenshot_dir:
        folder_path_label.config(text=selected_screenshot_dir)
        Brint(f"[UI] Folder label set to {selected_screenshot_dir}")
    else:
        folder_path_label.config(text="Aucun dossier sélectionné")
        Brint("[UI] Folder label cleared")


def save_config():
    try:
        config_data["screenshot_dir"] = selected_screenshot_dir
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        Brint("[CONFIG] ✅ Configuration sauvegardée")
    except Exception as e:
        Brint("[CONFIG] ❌ Erreur lors de la sauvegarde :", str(e))


def render_tagged_transcription():
    try:
        Brint("[TAG FORMATTER] 🚀 Affichage avec insertion d'images alignées sur les timestamps")

        from tkinter import messagebox
        import os
        import subprocess

        def open_image(path):
            try:
                subprocess.Popen(f'explorer "{path}"')
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d’ouvrir l’image : {e}")

        tagged_text_widget.delete("1.0", tk.END) # Clear widget before populating

        tagged_text_widget.tag_config("black", foreground="black")
        tagged_text_widget.tag_config("blue", foreground="blue")
        tagged_text_widget.tag_config("green", foreground="darkgreen")
        tagged_text_widget.tag_config("orange", foreground="orange")
        tagged_text_widget.tag_config("purple", foreground="purple", underline=1)
        tagged_text_widget.tag_config("gray", foreground="gray", font=("Arial", 9, "italic"))

        full_text = transcription_text_widget.get("1.0", tk.END).strip()
        lines = full_text.splitlines()
        screenshots = get_screenshots_with_timestamps()

        try:
            word_timeline = render_tagged_transcription.word_timeline
        except AttributeError:
            Brint("[TAG FORMATTER] ⚠️ Aucun word_timeline défini, utilisant la transcription brute.")
            plain_text_content = transcription_text_widget.get("1.0", tk.END)
            tagged_text_widget.insert("1.0", plain_text_content)
            current_text_for_tango = tagged_text_widget.get("1.0", tk.END)
            if current_text_for_tango.endswith("\n"):
                current_text_for_tango = current_text_for_tango[:-1]

            insertion_indices_plain = format_tags_for_display(current_text_for_tango)
            Brint(f"[TAG FORMATTER] TANGO newline indices (plain text fallback): {insertion_indices_plain}")
            for index_plain in insertion_indices_plain:
                tk_index_plain = f"1.0+{index_plain}c"
                tagged_text_widget.insert(tk_index_plain, "\r\n")
            Brint("[TAG FORMATTER] ✅ Formattage TANGO (plain text fallback) terminé.")
            return

        next_image_index = 0
        word_index = 0

        for line in lines:
            words = line.strip().split()
            for word in words:
                word_clean = word.strip(".,:;!?(){}[]").lower()

                timestamp = None
                if word_index < len(word_timeline):
                    timestamp = word_timeline[word_index]["start"]
                else:
                    Brint(f"[TAG FORMATTER] ⚠️ Index {word_index} dépasse timeline ({len(word_timeline)} mots)")

                while next_image_index < len(screenshots):
                    img_ts, img_name = screenshots[next_image_index]
                    if timestamp is not None and abs(img_ts - timestamp) <= 1.0:
                        tag = f"img_{img_ts}"
                        full_path = os.path.join(selected_screenshot_dir, img_name)
                        tagged_text_widget.insert(tk.END, f"[IMG: {img_name}]\n", tag)
                        tagged_text_widget.tag_config(tag, foreground="purple", underline=1)
                        tagged_text_widget.tag_bind(tag, "<Button-1>", lambda e, p=full_path: open_image(p))
                        Brint(f"[TAG FORMATTER] 📷 Image insérée : {img_name} @ {img_ts:.2f}s")
                        next_image_index += 1
                    elif timestamp is not None and img_ts < timestamp - 1.0:
                        next_image_index += 1
                    else:
                        break

                if word_clean in action_keywords:
                    color = action_keywords[word_clean]
                    tagged_text_widget.insert(tk.END, f"[{word_clean.upper()}] ", color)
                elif word_clean in target_keywords:
                    color = target_keywords[word_clean]
                    tagged_text_widget.insert(tk.END, f"[{word_clean.upper()}] ", color)
                else:
                    tagged_text_widget.insert(tk.END, word + " ", "black")

                word_index += 1
            tagged_text_widget.insert(tk.END, "\n")

        current_text = tagged_text_widget.get("1.0", tk.END)
        if current_text.endswith("\n"):
            current_text = current_text[:-1]

        insertion_indices = format_tags_for_display(current_text)

        Brint(f"[TAG FORMATTER] TANGO newline indices (reverse sorted): {insertion_indices}")

        for index in insertion_indices:
            tk_index = f"1.0+{index}c"
            tagged_text_widget.insert(tk_index, "\r\n")

        Brint("[TAG FORMATTER] ✅ Formattage terminé avec images.")

    except Exception as e:
        Brint("[TAG FORMATTER] ❌ ERREUR :", str(e))
        tagged_text_widget.insert(tk.END, "\n---\n[TAGS EXPLICITES MENU]\n", "gray")
        pattern = re.compile(r"\[MENU\]\s*(\w+)\s+([\wàâäéèêëïîôöùûüç\'\- ]+)", re.IGNORECASE)

        clean_text = tagged_text_widget.get("1.0", tk.END)
        matches = pattern.findall(clean_text)
        for direction, label in matches:
            line = f"MENU {direction.upper()} {label.strip().title()}"
            tagged_text_widget.insert(tk.END, line + "\n", "blue")
            Brint(f"[TAG FORMATTER] 🧷 Ligne tag extraite : {line}")


def parse_text_widget_into_display_data():
    Brint("[TEXT → DATA] Extraction du texte depuis le widget")
    lines = transcription_text_widget.get("1.0", tk.END).strip().split()
    transcription_display_data.clear()
    for word in lines:
        transcription_display_data.append({
            "text": word,
            "confidence": 1.0  # on simule une confiance max
        })
    Brint(f"[TEXT → DATA] {len(transcription_display_data)} mots régénérés")



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

def select_wav_and_transcribe():
    """Open a dialog to choose a WAV file and transcribe it."""
    wav_path = filedialog.askopenfilename(
        title="Choisir un fichier WAV",
        filetypes=[("Fichiers WAV", "*.wav")]
    )
    if wav_path:
        Brint("[TRANSCRIBE] [FILE SELECTED]", wav_path)
        transcribe_file(wav_path)
        if send_to_openai_var and send_to_openai_var.get():
            text = transcription_text_widget.get("1.0", tk.END)
            send_transcription_to_chatgpt(text, wav_path)

def select_wav_and_transcribe_chunked():
    """Choose a WAV file and transcribe it in 5 min chunks."""
    wav_path = filedialog.askopenfilename(
        title="Choisir un fichier WAV",
        filetypes=[("Fichiers WAV", "*.wav")]
    )
    if wav_path:
        Brint("[CHUNK TRANSCRIBE] [FILE SELECTED]", wav_path)
        out_path = os.path.splitext(wav_path)[0] + "_chunked.txt"
        use_faster = use_faster_var.get() if use_faster_var else False
        transcribe_wav_in_chunks(wav_path, output_path=out_path, use_faster=use_faster)
        messagebox.showinfo("Transcription", f"Transcript saved to {out_path}")
        if send_to_openai_var and send_to_openai_var.get():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    text = f.read()
                send_transcription_to_chatgpt(text, wav_path)
            except Exception as e:
                Brint("[OPENAI] Error reading transcript:", str(e))

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

def on_slider_release(event):
    value = confidence_threshold.get()
    Brint(f"[UI] [SLIDER RELEASED] Valeur = {value}")
    update_confidence_display(str(value))




def jump_to_time(timestamp_seconds):
    Brint(f"[NAV] [JUMP TO] Clic sur mot → {timestamp_seconds:.2f}s")

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

def select_directory():
    global selected_screenshot_dir
    selected_screenshot_dir = filedialog.askdirectory()
    if selected_screenshot_dir:
        Brint("[NAV] [DIR SELECTED] Watching folder:", selected_screenshot_dir)
        save_config()
        update_folder_path_label()
def open_folder_path():
    import os
    import subprocess
    if selected_screenshot_dir and os.path.exists(selected_screenshot_dir):
        Brint(f"[NAV] Opening folder {selected_screenshot_dir}")
        subprocess.Popen(f'explorer "{selected_screenshot_dir}"')

# def select_directory(): # This function is duplicated, removing one
#     global selected_screenshot_dir
#     selected_screenshot_dir = filedialog.askdirectory()
#     Brint("[NAV] [DIR SELECTED] Watching folder:", selected_screenshot_dir)
#     save_config()


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
    Brint(f"[AUDIO] Callback {frames} frames")
    if status:
        Brint("[AUDIO] [WARNING]", status)
    recorded_frames.append(indata.copy())
import os
from session_data import SessionData, Word, Screenshot
from random import uniform



def generate_fake_session(save_path="output_sessions/fake_test_session"):
    from random import uniform
    from dataclasses import dataclass
    import shutil
    import os


    Brint("[FAKE SESSION] 🚧 Démarrage génération session factice")
    os.makedirs(save_path, exist_ok=True)
    Brint(f"[FAKE SESSION] 📁 Répertoire de session : {save_path}")

    tagged_text = [
        "TANGO MENU ROOT Accueil",
        "TANGO MENU DOWN Profil",
        "TANGO MENU DOWN Préférences",
        "TANGO MENU BACK",
        "TANGO MENU SIDE Paramètres",
        "TANGO MENU DOWN Aide",
        "TANGO MENU SIDE Boutique"
    ]
    Brint("[FAKE SESSION] 🏷️ Texte taggué simulé :")
    for line in tagged_text:
        Brint(f"  ➤ {line}")

    words = []
    word_timeline = []
    parsed_tags = []
    time_val = 3.0 # Renamed to avoid conflict with time module

    for i, line in enumerate(tagged_text):
        parts = line.split()
        for word in parts:
            start = round(time_val, 2)
            words.append(Word(text=word, start=start, confidence=0.9, color="black"))
            word_timeline.append({"word": word, "start": start})
            Brint(f"[FAKE SESSION]   ➕ Word: '{word}' @ {start}s")
            time_val += uniform(0.3, 0.7)

        if len(parts) >= 3 and parts[1] == "MENU":
            direction = parts[2]
            label = " ".join(parts[3:]) if len(parts) > 3 else direction
            label_ts = next((w["start"] for w in word_timeline if w["word"] == label or label.startswith(w["word"])), time_val)
            parsed_tags.append({
                "type": "MENU",
                "direction": direction,
                "label": label,
                "start": label_ts
            })
    from PIL import Image, ImageDraw, ImageFont

    def create_fake_screenshot(path, label, width=640, height=360):
        img = Image.new("RGB", (width, height), color=(40, 40, 40))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            font = None
        draw.text((20, height // 2 - 20), f"{label}", fill=(255, 255, 255), font=font)
        img.save(path)

    screenshots = []
    screenshots_dir = os.path.join(save_path, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    for tag in parsed_tags:
        t = round(tag["start"], 2)
        label = tag["label"]
        fname = f"{t:05.2f}_test.png"
        dest_path = os.path.join(screenshots_dir, fname)
        create_fake_screenshot(dest_path, label)
        screenshots.append(Screenshot(timestamp=t, filename=fname))
        Brint(f"[FAKE SESSION]   🖼️ Screenshot: {fname} @ {t:.2f}s")

    session = SessionData(
        session_id="fake_test_session",
        audio_path="dummy.wav"
    )
    session.words = words
    session.screenshots = screenshots
    Brint(f"[DEBUG] Nombre de screenshots dans la session : {len(screenshots)}")
    for s in screenshots:
        Brint(f"  🖼️ {s.filename} @ {s.timestamp}s")

    session.save(save_path)
    Brint("[SAVE] ✅ Session sauvegardée dans " + os.path.join(save_path, "fake_test_session.json"))

    transcription_text_widget.delete("1.0", tk.END)
    transcription_text_widget.insert("1.0", "\n".join([line.replace("[", "").replace("]", "") for line in tagged_text]))

    tagged_text_widget.delete("1.0", tk.END)
    tagged_text_widget.insert("1.0", "\n".join(tagged_text))
    Brint("[UI] Texte fake injecté dans les widgets 'Transcription' et 'Tags détectés'")

    return session, tagged_text, word_timeline, parsed_tags


def toggle_record():
    global recording, stream, recorded_frames, record_start_time, timer_running
    if not recording:
        recorded_frames = []
        record_start_time = time.time()
        timer_running = True
        update_timer()

        global CAPTURE_OUTPUT_AUDIO
        loopback = CAPTURE_OUTPUT_AUDIO
        Brint("[AUDIO] [SETUP] loopback =", loopback)

        try:
            device_to_use = SELECTED_LOOPBACK_DEVICE_INDEX if loopback else sd.default.device[0] # Default input if not loopback
            if loopback and SELECTED_LOOPBACK_DEVICE_INDEX is None:
                 Brint("[AUDIO] [ERROR] Aucun périphérique sélectionné pour loopback.")
                 messagebox.showerror("Erreur Audio", "Aucun périphérique de sortie (loopback) n'a été sélectionné.")
                 return

            stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                device=device_to_use,
                loopback=loopback if SELECTED_LOOPBACK_DEVICE_INDEX is not None else False # Ensure loopback is False if no device
            )
            stream.start()
            recording = True
            Brint("[AUDIO] [RECORDING] ➤ Démarrage enregistrement")
            record_button.config(text="⏹ Stop + Transcribe")

        except Exception as e:
            Brint("[AUDIO] [ERROR] Échec de la configuration du stream :", str(e))
            messagebox.showerror("Erreur Audio", f"Impossible de démarrer l'enregistrement : {e}")
            return
    else:
        if stream:
            stream.stop()
            stream.close()
        recording = False
        timer_running = False

        record_button.config(text="⏺ Start Recording")
        timestamp_val = time.strftime("%Y%m%d-%H%M%S") # Renamed
        wav_path = os.path.join(os.path.expanduser("~"), "Videos", f"capture_{timestamp_val}.wav")
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 2 bytes for int16
            wf.setframerate(SAMPLERATE)
            wf.writeframes((np.concatenate(recorded_frames) * 32767).astype(np.int16).tobytes())
        Brint("[AUDIO] [RECORDING] ✅ Enregistrement terminé :", wav_path)
        transcribe_file(wav_path)


def diarize_speakers(wav_path):
    """Return list of speaker segments using pyannote pipeline."""
    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        Brint("[DIARIZATION] pyannote.audio not available:", str(exc))
        return []

    try:
        token = None
        if os.path.exists("secret.js"):
            Brint("[DIARIZATION] Reading token from secret.js")
            with open("secret.js", "r") as f:
                secret_content = f.read()
            match = re.search(r"token\s*[:=]\s*['\"](.+?)['\"]", secret_content)
            if match:
                token = match.group(1).strip()
                preview = token[:4] + "..." + token[-4:] if len(token) > 8 else token
                Brint(f"[DIARIZATION] Token detected ({len(token)} chars): {preview}")
            else:
                Brint("[DIARIZATION] Token pattern not found in secret.js")
        else:
            Brint("[DIARIZATION] secret.js not found")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=token or "token"
        )
        Brint("[DIARIZATION] Pipeline initialized, processing", wav_path)
        diarization = pipeline(wav_path)
        Brint("[DIARIZATION] Pipeline processing completed")
    except Exception as exc:
        Brint("[DIARIZATION] Error during diarization:", str(exc))
        return []

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        Brint(
            f"[DIARIZATION] Segment {speaker}: {turn.start:.2f}s -> {turn.end:.2f}s"
        )
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    Brint(f"[DIARIZATION] Detected {len(segments)} segments")
    return segments



def transcribe_file(wav_path):
    from session_data import SessionData, Word, Screenshot # Moved import
    global last_transcribed_wav_path
    last_transcribed_wav_path = wav_path
    Brint("[TRANSCRIBE] [TRANSCRIPTION STARTED]", wav_path)

    transcription_notebook.select(transcription_tab_frame)
    transcription_text_widget.delete("1.0", tk.END)

    confidence_index.clear()
    confidence_index_list.delete(0, tk.END)

    session = SessionData(
        session_id=os.path.basename(wav_path).replace(".wav", ""),
        audio_path=wav_path
    )

    if diarization_var and diarization_var.get():
        Brint("[DIARIZATION] Speaker identification enabled")
        speaker_segments = diarize_speakers(wav_path)
        Brint(f"[DIARIZATION] Speaker segments: {speaker_segments}")
    else:
        Brint("[DIARIZATION] Skipping speaker identification")
        speaker_segments = []

    def get_speaker(timestamp: float):
        for seg in speaker_segments:
            if seg["start"] <= timestamp <= seg["end"]:
                return seg["speaker"]
        return None

    def insert_word_to_widget(word_text, start, conf): # Renamed for clarity
        color = "black" if conf > 0.8 else "orange" if conf > 0.5 else "red"
        tag = f"word_{start:.2f}"

        transcription_text_widget.insert(tk.END, word_text + " ", (tag,))
        transcription_text_widget.tag_config(tag, foreground=color)
        transcription_text_widget.tag_bind(tag, "<Button-1>", lambda e, t=start: jump_to_time(t))

        if color == "red": # Assuming low confidence words are added to index
            confidence_index[f"{word_text}_{start}"] = tag # More unique key
            confidence_index_list.insert(tk.END, f"{word_text} ({start:.2f}s)")

        speaker = get_speaker(start)
        session.words.append(Word(text=word_text, start=start, confidence=conf, color=color, speaker=speaker))

    use_faster = use_faster_var.get() if use_faster_var else False
    if use_faster:
        Brint("[TRANSCRIBE] [MODE] ➤ Using faster-whisper (CUDA forcé)")
        try:
            import torch
            from faster_whisper import WhisperModel

            # FORCE GPU + FLOAT16
            device = "cuda"
            compute_type = "float16"

            Brint(f"[FASTER] Initialisation du modèle sur : {device} ({compute_type})")

            faster_model = WhisperModel("base", device=device, compute_type=compute_type)
            segments, info = faster_model.transcribe(
                wav_path, language="fr", beam_size=5, word_timestamps=True
            )

            current_word_timeline = []
            for segment in segments:
                for word_obj in segment.words:
                    word_text = word_obj.word
                    start = word_obj.start
                    conf = word_obj.probability or 1.0
                    if conf < confidence_threshold.get():
                        continue
                    insert_word_to_widget(word_text, start, conf)
                    current_word_timeline.append({"word": word_text, "start": start})
            render_tagged_transcription.word_timeline = current_word_timeline

        except Exception as e:
            Brint("[TRANSCRIBE] [ERROR - FASTER]", str(e))
            transcription_text_widget.insert(tk.END, f"[ERREUR FASTER-WHISPER] {str(e)}\n")
            return


    else:
        Brint("[TRANSCRIBE] [MODE] ➤ Using classic Whisper (CUDA forcé)")
        try:
            import whisper
            model_gpu = whisper.load_model("base", device="cuda")  # ← GPU forcé ici

            result = model_gpu.transcribe(
                wav_path,
                language="fr",
                fp16=True,  # ← float16 (accélération sur GPU)
                temperature=0,
                beam_size=5,
                word_timestamps=True
            )

            current_word_timeline = []
            segments = result.get("segments", [])
            for seg in segments:
                for word_info in seg["words"]:
                    word = word_info["word"]
                    start = word_info["start"]
                    conf = word_info.get("probability", 1.0)
                    if conf < confidence_threshold.get():
                        continue
                    insert_word_to_widget(word, start, conf)
                    current_word_timeline.append({"word": word, "start": start})
            render_tagged_transcription.word_timeline = current_word_timeline

        except Exception as e:
            Brint("[TRANSCRIBE] [ERROR - WHISPER CLASSIC]", str(e))
            transcription_text_widget.insert(tk.END, f"[ERREUR Whisper classique] {str(e)}\n")
            return



    render_tagged_transcription() # Call after transcription is done and timeline is set

    for seconds, fname in get_screenshots_with_timestamps():
        session.screenshots.append(Screenshot(filename=fname, timestamp=seconds))

    session.save("output_sessions/")
    from session_view_generator import generate_session_view
    generate_session_view(session)

    # Write plain text transcript next to the WAV file
    txt_out = os.path.splitext(wav_path)[0] + "_transcription.txt"
    try:
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(transcription_text_widget.get("1.0", tk.END).strip())
        Brint(f"[SAVE] \u2705 Transcription texte sauvegard\u00e9e : {txt_out}")
    except Exception as e:
        Brint("[SAVE] [ERROR TXT]", str(e))

    if send_to_openai_var and send_to_openai_var.get():
        transcript_text = transcription_text_widget.get("1.0", tk.END)
        send_transcription_to_chatgpt(transcript_text, wav_path)

def send_transcription_to_chatgpt(text: str, wav_path: str) -> None:
    """Send *text* to OpenAI ChatGPT and save the response next to *wav_path*."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            Brint("[OPENAI] No API key found in OPENAI_API_KEY")
            return
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
        )
        answer = resp.choices[0].message["content"]
        out_path = os.path.splitext(wav_path)[0] + "_chatgpt.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(answer)
        Brint(f"[OPENAI] Réponse sauvegardée : {out_path}")
    except Exception as e:
        Brint("[OPENAI] Error:", str(e))

def update_timer():
    global timer_running, record_start_time, timer_label
    if timer_running:
        elapsed = int(time.time() - record_start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        timer_label.config(text=f"Durée : {minutes:02}:{seconds:02}")
        timer_label.after(1000, update_timer)


def launch_gui():
    global timer_label, audio_output_var, record_button, use_faster_var, confidence_threshold, diarization_var
    global transcription_notebook, confidence_index_tab, confidence_index_list, confidence_index
    global transcription_tab_frame, transcription_text_widget, tagged_tab_frame, tagged_text_widget
    global folder_path_label, last_word_timeline, send_to_openai_var

    last_word_timeline = []


    def toggle_output_audio():
        global CAPTURE_OUTPUT_AUDIO
        CAPTURE_OUTPUT_AUDIO = audio_output_var.get() # Directly use var's value
        Brint(f"[AUDIO] [CONFIG] Capture sortie audio = {CAPTURE_OUTPUT_AUDIO}")

    def toggle_faster():
        Brint(f"[TRANSCRIBE] [OPTION] Utiliser faster-whisper = {use_faster_var.get()}")

    def toggle_diarization():
        Brint(f"[DIARIZATION] [OPTION] Identifier les speakers = {diarization_var.get()}")

    def update_confidence_display_on_slider(val_str): # Renamed to avoid conflict
        val = float(val_str)
        Brint(f"[TRANSCRIBE] [FILTER UPDATE] Nouveau seuil de confiance = {val}")
        if last_transcribed_wav_path:
            Brint("[TRANSCRIBE] [FILTER UPDATE] Relance affichage transcription filtrée")
            # Temporarily disable transcription call to avoid loop if transcribe_file calls render_tagged_transcription
            # which might call this again if not careful
            # transcribe_file(last_transcribed_wav_path)
            # Instead, directly re-render if possible, or ensure transcribe_file doesn't auto-trigger this.
            # For now, a simple Brint to show intention:
            Brint(f"[TRANSCRIBE] [FILTER UPDATE] Would re-filter display for {last_transcribed_wav_path} with threshold {val}")
            # Actual re-filtering needs careful implementation to avoid re-transcribing fully.
            # It should ideally just hide/show words based on new threshold.
        else:
            Brint("[TRANSCRIBE] [FILTER UPDATE] Aucun fichier à réafficher")


    def on_slider_value_release(event): # Renamed from on_slider_release
        value = confidence_threshold.get()
        Brint(f"[UI] [SLIDER RELEASED] Valeur = {value}")
        update_confidence_display_on_slider(str(value))

    def reset_confidence_slider(event): # Renamed
        Brint("[UI] [SLIDER RESET] Double-clic → remise à 0.0")
        confidence_threshold.set(0.0)
        update_confidence_display_on_slider("0.0")

    def start_all_processes(): # Renamed
        if not selected_screenshot_dir:
            messagebox.showerror("Erreur", "Aucun dossier de captures d'écran n'a été sélectionné.")
            Brint("[NAV] [ERROR] Aucun dossier sélectionné.")
            return
        start_watching_directory()
        Brint("[NAV] [ACTION] ▶ Lancement du processus complet (audio + screenshot)")
        # toggle_record() # Optionally auto-start recording

    def on_confidence_list_item_click(event): # Renamed
        selection = confidence_index_list.curselection()
        if not selection: return

        selected_item_text = confidence_index_list.get(selection[0])
        # Extract word and start time if stored in a specific format e.g. "word (start_time)"
        match_item = re.match(r"^(.*?) \((\d+\.\d+)s\)$", selected_item_text)
        if not match_item:
            Brint(f"[INDEX] Could not parse item: {selected_item_text}")
            return

        word_text = match_item.group(1)
        start_time_str = match_item.group(2)
        tag_key = f"{word_text}_{start_time_str}" # Reconstruct key used for confidence_index

        tag = confidence_index.get(tag_key)

        if tag:
            Brint(f"[INDEX] Mot sélectionné : '{word_text}' Tag: {tag}")
            transcription_notebook.select(transcription_tab_frame)
            ranges = transcription_text_widget.tag_nextrange(tag, "1.0")
            if ranges:
                start_idx, end_idx = ranges
                transcription_text_widget.see(start_idx)
                transcription_text_widget.tag_remove("highlight", "1.0", tk.END)
                transcription_text_widget.tag_add("highlight", start_idx, end_idx)
                transcription_text_widget.tag_config("highlight", background="yellow", foreground="black") # Ensure visibility
            else:
                Brint(f"[INDEX] Tag '{tag}' introuvable pour '{word_text}'.")
        else:
            Brint(f"[INDEX] No tag found for key '{tag_key}' in confidence_index.")
    def check_or_prompt_openai_key():
        if load_openai_api_key_from_file():
            openai_key_status_label.config(text="✅ Clé API : chargée", fg="green")
        else:
            new_key = simpledialog.askstring("Clé API OpenAI", "Entre ta clé OpenAI :")
            if new_key:
                try:
                    with open(".openai_key", "w") as f:
                        f.write(new_key.strip())
                    openai.api_key = new_key.strip()
                    os.environ["OPENAI_API_KEY"] = new_key.strip()
                    openai_key_status_label.config(text="✅ Clé API : chargée", fg="green")
                except Exception as e:
                    Brint("[OPENAI] Erreur de sauvegarde :", e)
            else:
                openai_key_status_label.config(text="❌ Clé API non définie", fg="red")



    def on_generate_fake_menu_tree_clicked(): # Renamed
        Brint("[MENU] ▶ Génération arborescence FAKE pour test")
        session, tagged_text_list, word_timeline_list, parsed_tags_list = generate_fake_session() # Ensure names are distinct
        global last_loaded_session
        last_loaded_session = session

        # Use the returned lists directly
        tree = build_menu_tree_from_tagged_text(tagged_text_list, word_timeline_list, screenshots=session.screenshots, parsed_tags=parsed_tags_list)

        output_session_dir = "output_sessions/fake_test_session" # Renamed
        base_path = os.path.expanduser("~")  # ton dossier utilisateur, ex: C:\Users\Vincent B
        output_folder = os.path.join(base_path, "Documents", "output_sessions")
        os.makedirs(output_folder, exist_ok=True)

        global last_word_timeline # Ensure this is the correct timeline to use
        last_word_timeline = word_timeline_list # Assign the timeline from fake session
        render_tagged_transcription.word_timeline = last_word_timeline # Set for current rendering pass

        import json # Local import
        with open(os.path.join(output_session_dir, "menu_tree.json"), "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)

        from menu_html_utils import generate_menu_tree_html # Local import
        html_path = os.path.join(output_session_dir, "menu_tree.html")
        generate_menu_tree_html(tree, html_path)

        import webbrowser # Local import
        webbrowser.open(f"file://{os.path.abspath(html_path)}")

    load_config()
    root = tk.Tk()
    root.title("Live Screenshot Annotator")

    timer_label = tk.Label(root, text="Durée : 00:00")
    timer_label.pack()

    tk.Label(root, text="Sélectionne le dossier de captures d'écran :").pack(pady=10)
    tk.Button(root, text="Choisir dossier", command=select_directory).pack(pady=5)
    folder_path_label = tk.Label(root, text="Aucun dossier sélectionné", fg="blue", cursor="hand2")
    folder_path_label.pack()
    folder_path_label.bind("<Button-1>", lambda e: open_folder_path())
    openai_key_status_label = tk.Label(root, text="🔒 Clé API : non chargée", fg="red")
    openai_key_status_label.pack()
    # Mise à jour visuelle immédiate du statut de la clé API
    key = os.getenv("OPENAI_API_KEY", "")
    if key.startswith("sk-"):
        openai_key_status_label.config(text="✅ Clé API : chargée", fg="green")
    else:
        openai_key_status_label.config(text="🔒 Clé API : non chargée", fg="red")

    tk.Button(root, text="🔑 Vérifier / définir clé OpenAI", command=check_or_prompt_openai_key).pack(pady=5)

    tk.Button(root, text="▶ Démarrer Annotation & Audio", command=start_all_processes).pack(pady=20) # Renamed command
    tk.Button(root, text="🎛 Choisir sortie audio (loopback)", command=choose_loopback_device).pack(pady=5)

    audio_output_var = tk.BooleanVar(value=CAPTURE_OUTPUT_AUDIO) # Init with global
    tk.Checkbutton(root, text="🎧 Capturer aussi le son de sortie (loopback)", variable=audio_output_var, command=toggle_output_audio).pack()

    use_faster_var = tk.BooleanVar(value=False) # Default to False
    tk.Checkbutton(root, text="⚡ Utiliser Faster-Whisper (GPU optimisé)", variable=use_faster_var, command=toggle_faster).pack()

    diarization_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="🗣 Identifier les speakers", variable=diarization_var, command=toggle_diarization).pack()

    send_to_openai_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="💬 Envoyer la transcription à ChatGPT", variable=send_to_openai_var).pack()

    record_button = tk.Button(root, text="⏺ Start Recording", command=toggle_record)
    record_button.pack(pady=10)
    tk.Button(root, text="🎙️ Transcrire un fichier WAV", command=select_wav_and_transcribe).pack(pady=5)
    tk.Button(root, text="📃 Transcription longue (5 min chunks)", command=select_wav_and_transcribe_chunked).pack(pady=5)
    tk.Button(root, text="🧪 Générer un scénario de test FAKE", command=on_generate_fake_menu_tree_clicked, bg="#ffeecc").pack(pady=10) # Renamed command

    confidence_threshold = tk.DoubleVar(value=0.0)
    tk.Label(root, text="Seuil de confiance (affichage):").pack()
    slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=confidence_threshold)
    slider.pack()
    slider.bind("<ButtonRelease-1>", on_slider_value_release) # Renamed command
    slider.bind("<Double-Button-3>", reset_confidence_slider) # Renamed command, and typically Button-3 is right-click

    transcription_notebook = ttk.Notebook(root)
    transcription_tab_frame = tk.Frame(transcription_notebook)
    transcription_text_widget = tk.Text(transcription_tab_frame, wrap=tk.WORD)
    transcription_text_widget.pack(fill="both", expand=True)
    tk.Button(transcription_tab_frame, text="📂 Load Transcription", command=load_transcription_from_json).pack(side=tk.LEFT, pady=5, padx=5) # Adjusted packing
    tk.Button(transcription_tab_frame, text="💾 Save Transcription", command=save_transcription_to_json).pack(side=tk.LEFT, pady=5, padx=5) # Adjusted packing
    transcription_notebook.add(transcription_tab_frame, text="📝 Transcription")

    tagged_tab_frame = tk.Frame(transcription_notebook)
    tagged_text_content_frame = tk.Frame(tagged_tab_frame) # Renamed frame for clarity
    tagged_text_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5) # expand True

    tagged_scrollbar = tk.Scrollbar(tagged_text_content_frame, orient=tk.VERTICAL)
    tagged_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tagged_text_widget = tk.Text(tagged_text_content_frame, wrap=tk.WORD, height=10, yscrollcommand=tagged_scrollbar.set)
    tagged_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) # expand True
    tagged_scrollbar.config(command=tagged_text_widget.yview)
    tk.Button(tagged_tab_frame, text="🌀 Regénérer HTML", command=lambda: on_regenerate_html_from_tags()).pack(pady=5) # Used lambda for simplicity
    transcription_notebook.add(tagged_tab_frame, text="📌 Tags détectés")

    confidence_index_tab = tk.Frame(transcription_notebook)
    confidence_index_list = tk.Listbox(confidence_index_tab)
    confidence_index_list.pack(fill="both", expand=True)
    confidence_index_list.bind("<Double-Button-1>", on_confidence_list_item_click) # Renamed command
    transcription_notebook.add(confidence_index_tab, text="🔍 Confidence Index")

    transcription_notebook.pack(fill="both", expand=True, padx=10, pady=10)

    def tab_changed_handler(event): # Renamed
        selected_tab_text = transcription_notebook.tab(transcription_notebook.select(), "text")
        Brint(f"[UI] [TAB SWITCH] Onglet sélectionné = {selected_tab_text}")
        if selected_tab_text == "📌 Tags détectés":
            Brint("[UI] [TAB SWITCH] ➤ Lancement render_tagged_transcription()")
            render_tagged_transcription()
    transcription_notebook.bind("<<NotebookTabChanged>>", tab_changed_handler) # Renamed command

    root.mainloop()
    check_or_prompt_openai_key()
    # Vérification UI après le chargement
    # # if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"].startswith("sk-"):
        # # openai_key_status_label.config(text="✅ Clé API : chargée", fg="green")
    # # else:
        # # openai_key_status_label.config(text="🔒 Clé API : non chargée", fg="red")


# move_console_to_right_half() # Consider if this is needed or part of specific dev setup
launch_gui()