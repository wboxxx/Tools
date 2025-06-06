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


action_keywords = {
    "click": "blue",
    "slide": "blue",
    "open": "blue",
    "close": "blue",
    "select": "blue",
    "comment": "orange"
}

target_keywords = {
    "bouton": "green",
    "menu": "green",
    "onglet": "green",
    "carrousel": "green",
    "bandeau": "green",
    "popup": "green",
    "champ": "green",
    "valider": "green",
    "retour": "green"
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
