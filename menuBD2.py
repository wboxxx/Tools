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

timer_label = None
record_start_time = None
timer_running = False
confidence_threshold = None
last_transcribed_wav_path = None
confidence_index = {}  # clé = mot, valeur = (tab_frame, tag)

def format_tags_for_display(raw_text: str) -> str:
    # Ajoute un retour à la ligne avant chaque 'TANGO' (sauf si c'est déjà en début de ligne)
    import re
    # Ajoute \n avant 'TANGO' précédé d'un caractère autre que \n ou début de chaîne
    formatted = re.sub(r'(?<!^)(?<!\n)(TANGO)', r'\r\1', raw_text)
    return formatted





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
    return sorted(screenshots)


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
    else:
        folder_path_label.config(text="Aucun dossier sélectionné")


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

        tagged_text_widget.delete("1.0", tk.END)
        tagged_text_widget.tag_config("black", foreground="black")
        tagged_text_widget.tag_config("blue", foreground="blue")
        tagged_text_widget.tag_config("green", foreground="darkgreen")
        tagged_text_widget.tag_config("orange", foreground="orange")
        tagged_text_widget.tag_config("purple", foreground="purple", underline=1)


        full_text = transcription_text_widget.get("1.0", tk.END).strip()
        lines = full_text.splitlines()
        screenshots = get_screenshots_with_timestamps()  # -> list of (seconds, filename)

        # timestamps mot à mot reconstruits depuis la transcription d’origine
        try:
            word_timeline = render_tagged_transcription.word_timeline
        except AttributeError:
            Brint("[TAG FORMATTER] ⚠️ Aucun word_timeline défini")
            return
        next_image_index = 0

        word_index = 0
        for line in lines:
            words = line.strip().split()

            for word in words:
                word_clean = word.strip(".,:;!?(){}[]").lower()

                # Chercher un timestamp associé
                timestamp = None
                if word_index < len(word_timeline):
                    timestamp = word_timeline[word_index]["start"]
                else:
                    Brint(f"[TAG FORMATTER] ⚠️ Index {word_index} dépasse timeline")

                # ➕ Insertion d'image si une image est proche de ce timestamp
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
                        next_image_index += 1  # trop ancien, skip
                    else:
                        break  # image future, on attend

                # ➕ Affichage du mot (formaté si reconnu)
                if word_clean in action_keywords:
                    color = action_keywords[word_clean]
                    tagged_text_widget.insert(tk.END, f"[{word.upper()}] ", color)
                elif word_clean in target_keywords:
                    color = target_keywords[word_clean]
                    tagged_text_widget.insert(tk.END, f"[{word.upper()}] ", color)
                else:
                    tagged_text_widget.insert(tk.END, word + " ", "black")

                word_index += 1

            tagged_text_widget.insert(tk.END, "\n")

        # Couleurs
        tagged_text_widget.tag_config("black", foreground="black")
        tagged_text_widget.tag_config("blue", foreground="blue")
        tagged_text_widget.tag_config("green", foreground="darkgreen")
        tagged_text_widget.tag_config("orange", foreground="orange")
        tagged_text_widget.tag_config("purple", foreground="purple", underline=1)
        tagged_text_widget.tag_config("gray", foreground="gray", font=("Arial", 9, "italic"))

        formatted_text = format_tags_for_display(raw_text)
        tagged_text_widget.delete("1.0", tk.END)
        tagged_text_widget.insert("1.0", formatted_text)

        Brint("[TAG FORMATTER] ✅ Formattage terminé avec images.")

    except Exception as e:
        Brint("[TAG FORMATTER] ❌ ERREUR :", str(e))
        # ➕ Extraire les lignes de tag [MENU] pour visualisation claire
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
        subprocess.Popen(f'explorer "{selected_screenshot_dir}"')

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

    # Texte simulé
    
    # Texte simulé avec BACK
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
    time = 3.0

    for i, line in enumerate(tagged_text):
        parts = line.split()
        for word in parts:
            start = round(time, 2)
            words.append(Word(text=word, start=start, confidence=0.9, color="black"))
            word_timeline.append({"word": word, "start": start})
            Brint(f"[FAKE SESSION]   ➕ Word: '{word}' @ {start}s")
            time += uniform(0.3, 0.7)

        if len(parts) >= 3 and parts[1] == "MENU":
            direction = parts[2]
            label = " ".join(parts[3:]) if len(parts) > 3 else direction  # BACK has no label
            label_ts = next((w["start"] for w in word_timeline if w["word"] == label or label.startswith(w["word"])), time)
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
            font = None  # fallback sans font personnalisée
        draw.text((20, height // 2 - 20), f"{label}", fill=(255, 255, 255), font=font)
        img.save(path)

    screenshots = []
    screenshots_dir = os.path.join(save_path, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    for tag in parsed_tags:
        t = round(tag["start"], 2)

        label = tag["label"]
        fname = f"{t:05.2f}_test.png"  # ex: "04.30_test.png"
        dest_path = os.path.join(screenshots_dir, fname)
        create_fake_screenshot(dest_path, label)
        screenshots.append(Screenshot(timestamp=t, filename=fname))
        Brint(f"[FAKE SESSION]   🖼️ Screenshot: {fname} @ {t:.2f}s")


    # Création session
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

    # Injection dans UI
    transcription_text_widget.delete("1.0", tk.END)
    transcription_text_widget.insert("1.0", "\n".join([line.replace("[", "").replace("]", "") for line in tagged_text]))

    tagged_text_widget.delete("1.0", tk.END)
    tagged_text_widget.insert("1.0", "\n".join(tagged_text))
    Brint("[UI] Texte fake injecté dans les widgets 'Transcription' et 'Tags détectés'")

    return session, tagged_text, word_timeline, parsed_tags


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
    from session_data import SessionData, Word, Screenshot
    global last_transcribed_wav_path
    last_transcribed_wav_path = wav_path
    Brint("[TRANSCRIBE] [TRANSCRIPTION STARTED]", wav_path)

    transcription_notebook.select(transcription_tab_frame)
    transcription_text_widget.delete("1.0", tk.END)

    confidence_index.clear()
    confidence_index_list.delete(0, tk.END)

    # Init session
    session = SessionData(
        session_id=os.path.basename(wav_path).replace(".wav", ""),
        audio_path=wav_path
    )

    def insert_word(word_text, start, conf):
        color = "black" if conf > 0.8 else "orange" if conf > 0.5 else "red"
        tag = f"word_{start:.2f}"

        transcription_text_widget.insert(tk.END, word_text + " ", (tag,))
        transcription_text_widget.tag_config(tag, foreground=color)
        transcription_text_widget.tag_bind(tag, "<Button-1>", lambda e, t=start: jump_to_time(t))

        if color == "red":
            confidence_index[word_text] = tag
            confidence_index_list.insert(tk.END, word_text)

        session.words.append(Word(text=word_text, start=start, confidence=conf, color=color))

    # Transcription avec Faster-Whisper
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
                    if conf < confidence_threshold.get():
                        continue
                    insert_word(word_text, start, conf)

            render_tagged_transcription.word_timeline = [
                {"word": w.word, "start": w.start}
                for segment in segments for w in segment.words
            ]

        except Exception as e:
            Brint("[TRANSCRIBE] [ERROR - FASTER]", str(e))
            transcription_text_widget.insert(tk.END, f"[ERREUR FASTER-WHISPER] {str(e)}\n")
            return

        render_tagged_transcription()
    
    else:
        # Transcription classique Whisper
        Brint("[TRANSCRIBE] [MODE] ➤ Using classic Whisper")
        try:
            result = model.transcribe(
                wav_path,
                language="fr",
                fp16=False,
                temperature=0,
                beam_size=5,
                word_timestamps=True
            )

            segments = result.get("segments", [])
            for seg in segments:
                for word_info in seg["words"]:
                    word = word_info["word"]
                    start = word_info["start"]
                    conf = word_info.get("probability", 1.0)
                    if conf < confidence_threshold.get():
                        continue
                    insert_word(word, start, conf)

            render_tagged_transcription.word_timeline = [
                {"word": w["word"], "start": w["start"]}
                for seg in segments for w in seg["words"]
            ]

        except Exception as e:
            Brint("[TRANSCRIBE] [ERROR]", str(e))
            transcription_text_widget.insert(tk.END, f"[ERREUR Whisper classique] {str(e)}\n")
            return

        render_tagged_transcription()

    # Ajout des screenshots à la session
    for seconds, fname in get_screenshots_with_timestamps():
        session.screenshots.append(Screenshot(filename=fname, timestamp=seconds))

    # Sauvegarde finale
    session.save("output_sessions/")
    from session_view_generator import generate_session_view
    generate_session_view(session)
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
    global tagged_tab_frame
    global tagged_text_widget

    global folder_path_label
    global last_word_timeline
    last_word_timeline = []


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

    def on_generate_fake_menu_tree():
        Brint("[MENU] ▶ Génération arborescence FAKE pour test")

        session, tagged_text, word_timeline, parsed_tags = generate_fake_session()
        global last_loaded_session
        last_loaded_session = session

        
        tree = build_menu_tree_from_tagged_text(tagged_text, word_timeline, screenshots=session.screenshots, parsed_tags=parsed_tags)
        screenshots = [(s.timestamp, s.filename) for s in session.screenshots]

        tree = build_menu_tree_from_tagged_text(
            tagged_text,
            word_timeline,
            session.screenshots  # ✅ envoie les objets, pas des tuples
        )
        output_dir = "output_sessions/fake_test_session"
        os.makedirs(output_dir, exist_ok=True)
        last_word_timeline = word_timeline
        render_tagged_transcription.word_timeline = last_word_timeline

        # 💾 JSON
        import json
        with open(os.path.join(output_dir, "menu_tree.json"), "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)

        # 🌐 HTML
        from menu_html_utils import generate_menu_tree_html
        html_path = os.path.join(output_dir, "menu_tree.html")
        generate_menu_tree_html(tree, html_path)

        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(html_path)}")


   
    

    load_config()

    root = tk.Tk()
    root.title("Live Screenshot Annotator")

    # 🎛 UI : timer
    timer_label = tk.Label(root, text="Durée : 00:00")
    timer_label.pack()

    # 📂 Sélection du dossier
    tk.Label(root, text="Sélectionne le dossier de captures d'écran :").pack(pady=10)
    tk.Button(root, text="Choisir dossier", command=select_directory).pack(pady=5)
    folder_path_label = tk.Label(root, text="Aucun dossier sélectionné", fg="blue", cursor="hand2")
    folder_path_label.pack()
    folder_path_label.bind("<Button-1>", lambda e: open_folder_path())










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
    tk.Button(root, text="🧪 Générer un scénario de test FAKE", command=on_generate_fake_menu_tree, bg="#ffeecc").pack(pady=10)
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
    # Après transcription_text_widget.pack(...)
    transcription_text_widget.pack(fill="both", expand=True)
    load_button = tk.Button(transcription_tab_frame, text="📂 Load Transcription", command=load_transcription_from_json)
    load_button.pack(pady=5, anchor="ne")

    save_button = tk.Button(transcription_tab_frame, text="💾 Save Transcription", command=save_transcription_to_json)
    save_button.pack(pady=5, anchor="ne")

    transcription_notebook.add(transcription_tab_frame, text="📝 Transcription")
    transcription_notebook.select(transcription_tab_frame)

    
    transcription_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        # 📌 Onglet Tags détectés
    tagged_tab_frame = tk.Frame(transcription_notebook)
    # 📦 Frame contenant le widget texte + scrollbar
    tagged_text_frame = tk.Frame(tagged_tab_frame)
    tagged_text_frame.pack(fill=tk.X, padx=5, pady=5)

    tagged_scrollbar = tk.Scrollbar(tagged_text_frame, orient=tk.VERTICAL)
    tagged_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    tagged_text_widget = tk.Text(
        tagged_text_frame,
        wrap=tk.WORD,
        height=10,  # ← réduit la hauteur visible
        yscrollcommand=tagged_scrollbar.set
    )
    tagged_text_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
    tagged_scrollbar.config(command=tagged_text_widget.yview)

    def clean_tagged_lines(raw_text: str) -> List[str]:
        clean = raw_text.replace("[", "").replace("]", "")
        return [
            line.strip() for line in clean.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]



    # 🌀 Bouton de régénération HTML depuis les tags édités
    def on_regenerate_html_from_tags():
        Brint("[UI] 🔁 Regénération HTML à partir du contenu édité de 'Tags détectés'...")

        tagged_text_lines = clean_tagged_lines(tagged_text_widget.get("1.0", tk.END))

        if not last_loaded_session:
            Brint("[UI] ⚠️ Aucune session active pour accéder aux screenshots et timeline.")
            return

        screenshots = last_loaded_session.screenshots
        word_timeline = last_loaded_session.words
        session_path = last_loaded_session.path

        Brint(f"[UI] 📋 {len(tagged_text_lines)} lignes analysées :")
        for i, line in enumerate(tagged_text_lines):
            Brint(f"   {i+1:02d}: {line}")

        try:
            menu_tree = build_menu_tree_from_tagged_text(tagged_text_lines, word_timeline, screenshots)
            html_path = os.path.join(session_path, "menu_tree.html")
            generate_menu_tree_html(menu_tree, html_path)
            Brint(f"[UI] 🌐 Ouverture de {html_path}")
            webbrowser.open("file://" + os.path.abspath(html_path))
        except Exception as e:
            Brint(f"[UI] ❌ Erreur lors de la génération HTML : {e}")

    # Ajout du bouton UI
    regen_html_button = tk.Button(tagged_tab_frame, text="🌀 Regénérer HTML", command=on_regenerate_html_from_tags)
    regen_html_button.pack(pady=5)








    transcription_notebook.add(tagged_tab_frame, text="📌 Tags détectés")
    

    # 🔍 Onglet Confidence Index
    confidence_index_tab = tk.Frame(transcription_notebook)
    transcription_notebook.add(confidence_index_tab, text="🔍 Confidence Index")

    confidence_index_list = tk.Listbox(confidence_index_tab)
    confidence_index_list.pack(fill="both", expand=True)
    confidence_index_list.bind("<Double-Button-1>", on_confidence_word_click)
    Brint("[UI] [BIND] Bind appliqué sur Double-Click dans confidence_index_list")

    def on_tab_changed(event):
        selected_tab = transcription_notebook.tab(transcription_notebook.select(), "text")
        Brint(f"[UI] [TAB SWITCH] Onglet sélectionné = {selected_tab}")
        if selected_tab == "📌 Tags détectés":
            # Brint("[UI] [TAB SWITCH] ➤ Récupération du texte depuis le widget transcription")
            # full_text = transcription_text_widget.get("1.0", tk.END)
            Brint("[UI] [TAB SWITCH] ➤ Lancement render_tagged_transcription()")
            render_tagged_transcription()
    transcription_notebook.bind("<<NotebookTabChanged>>", on_tab_changed)
    Brint("[UI] [BIND] Bind sur changement d'onglet appliqué")

    # session, tagged_text, word_timeline = generate_fake_session()   
    # tree = build_menu_tree_from_tagged_text(
        # "\n".join(tagged_text),
        # word_timeline,
        # [(s.timestamp, s.filename) for s in session.screenshots]
    # )
    # import json
    # with open("output_sessions/fake_test_session/menu_tree.json", "w", encoding="utf-8") as f:
        # json.dump(tree, f, indent=2, ensure_ascii=False)
    # print_menu_tree(tree)


    # 🪟 Lancement de l'UI
    root.mainloop()

#move_console_to_right_half()

launch_gui()
