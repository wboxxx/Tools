# session_view_generator.py

from session_data import SessionData
import os
import datetime
from html import escape

# session_view_generator.py

from session_data import SessionData
import os
import datetime
from html import escape

def generate_session_view(session: SessionData, output_folder="session_views/"):
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{session.session_id}_view.html"
    filepath = os.path.join(output_folder, filename)

    def html_escape(text):
        return escape(str(text)).replace("\n", "<br>")

    def word_html(word):
        tooltip = f"{word.text} @ {word.start:.2f}s — conf: {word.confidence:.2f}"
        return f"<span class='word {word.color}' title='{tooltip}' data-timestamp='{word.start:.2f}'>{html_escape(word.text)}</span>"

    f = open(filepath, "w", encoding="utf-8")
    f.write("<!DOCTYPE html><html><head><meta charset='UTF-8'>\n")
    f.write(f"<title>Session View — {session.session_id}</title>\n")
    f.write("<style>\n")
    f.write("body { font-family: sans-serif; padding: 20px; }\n")
    f.write(".word { display: inline-block; margin: 2px; padding: 2px 5px; border-radius: 4px; cursor: pointer; }\n")
    f.write(".black { background-color: #f0f0f0; } .orange { background-color: #ffd699; } .red { background-color: #ffaaaa; }\n")
    f.write(".screenshot { margin: 20px 0; }\n")
    f.write("img { max-width: 500px; display: block; margin-bottom: 5px; }\n")
    f.write("h2 { border-bottom: 1px solid #ccc; margin-top: 30px; }\n")
    f.write(".timestamp { font-size: 0.9em; color: #666; }\n")
    f.write("</style>\n")

    f.write("<script>\n")
    f.write("""
    document.addEventListener("DOMContentLoaded", () => {
      document.querySelectorAll('.word').forEach(el => {
        el.addEventListener("click", () => {
          const t = parseFloat(el.dataset.timestamp);
          alert(`Jump to time: ${t.toFixed(2)}s`);
          // TODO: Hook up with video/audio player here
        });
      });
    });
    """)
    f.write("</script>\n")

    f.write("</head><body>\n")

    # ✅ HEADER
    f.write(f"<h1>Session View: {session.session_id}</h1>\n")
    f.write(f"<p><strong>Audio:</strong> {html_escape(session.audio_path)}</p>\n")
    f.write(f"<p><strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
    f.write(f"<p><strong>Mots détectés:</strong> {len(session.words)} | Captures: {len(session.screenshots)}</p>\n")

    # 🔠 Transcription
    f.write("<h2>Transcription horodatée</h2>\n<p>")
    sorted_words = sorted(session.words, key=lambda w: w.start)
    for word in sorted_words:
        f.write(word_html(word) + " ")
    f.write("</p>\n")

    # 🖼 Screenshots
    if session.screenshots:
        f.write("<h2>Captures d’écran</h2>\n")
        for s in sorted(session.screenshots, key=lambda s: s.timestamp):
            f.write("<div class='screenshot'>\n")
            f.write(f"<div class='timestamp'>⏱ {s.timestamp:.2f}s — {html_escape(s.filename)}</div>\n")
            f.write(f"<img src='../{s.filename}' alt='{s.filename}'>\n")
            f.write("</div>\n")

    f.write("</body></html>")
    f.close()

    print(f"[SESSION VIEW] ✅ Fichier généré : {filepath}")
    return filepath
