import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, BooleanVar

def convert_video(input_path, cut_to_2min, export_wav_only):
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in [".mkv", ".mp4", ".mov", ".m4v"]:
        messagebox.showerror("Erreur", "Format vidéo non supporté.")
        return

    output_base = os.path.splitext(input_path)[0]
    output_m4v = output_base + ".m4v"
    output_wav = output_base + ".wav"

    try:
        if export_wav_only:
            wav_command = [
                "ffmpeg", "-i", input_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2"
            ]
            if cut_to_2min:
                wav_command += ["-t", "00:02:00"]
            wav_command += [output_wav]
            subprocess.run(wav_command, check=True)
            messagebox.showinfo("Succès", f"Export audio terminé :\n{output_wav}")
        else:
            m4v_command = [
                "ffmpeg", "-i", input_path,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k"
            ]
            if cut_to_2min:
                m4v_command += ["-t", "00:02:00"]
            m4v_command += [output_m4v]
            subprocess.run(m4v_command, check=True)
            messagebox.showinfo("Succès", f"Conversion vidéo terminée :\n{output_m4v}")

    except subprocess.CalledProcessError:
        messagebox.showerror("Erreur", "Échec de la conversion")

def pick_file_and_convert():
    filepath = filedialog.askopenfilename(
        title="Choisir un fichier vidéo",
        filetypes=[("Fichiers vidéo", "*.mkv *.mp4 *.mov *.m4v")]
    )
    if filepath:
        convert_video(filepath, cut_var.get(), export_wav_var.get())

# UI setup
root = tk.Tk()
root.title("Convertisseur Vidéo ou Audio")
root.geometry("360x220")
root.resizable(False, False)

label = tk.Label(root, text="Convertir un fichier vidéo", font=("Arial", 12))
label.pack(pady=10)

cut_var = BooleanVar()
cut_checkbox = tk.Checkbutton(root, text="Limiter à 2 minutes", variable=cut_var)
cut_checkbox.pack()

export_wav_var = BooleanVar()
wav_checkbox = tk.Checkbutton(root, text="Exporter uniquement en WAV", variable=export_wav_var)
wav_checkbox.pack()

button = tk.Button(root, text="Choisir un fichier vidéo", command=pick_file_and_convert)
button.pack(pady=20)

root.mainloop()
