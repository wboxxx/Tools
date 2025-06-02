# session_view_generator.py

from session_data import SessionData
import os
import datetime
from html import escape

def generate_session_view(session: SessionData, output_folder="session_views/"):    os.makedirs(output_folder, exist_ok=True)
    filename = f"{session.session_id}_report.html"
    filepath = os.path.join(output_folder, filename)

    def html_escape(text):
        return escape(str(text)).replace("\n", "<br>")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='UTF-8'>\n")
        f.write(f"<title>Rapport — {session.session_id}</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: sans-serif; padding: 20px; }\n")
        f.write("h2 { border-bottom: 1px solid #ccc; }\n")
        f.write(".word { display: inline-block; margin: 2px; padding: 2px 6px; border-radius: 4px; }\n")
        f.write(".black { background-color: #e0e0e0; } .orange { background-color: #ffcc80; } .red { background-color: #ef9a9a; }\n")
        f.write(".screenshot { margin: 10px 0; }\n")
        f.write("</style>\n")
        f.write("</head><body>\n")

        # 🧾 Résumé
        f.write(f"<h1>Rapport : {session.session_id}</h1>\n")
        f.write(f"<p><strong>Audio :</strong> {html_escape(session.audio_path)}</p>\n")
        f.write(f"<p><strong>Date :</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
        f.write(f"<p><strong>Nombre de mots :</strong> {len(session.words)}</p>\n")
        f.write(f"<p><strong>Captures :</strong> {len(session.screenshots)}</p>\n")

        # 🔤 Transcription
        f.write("<h2>Transcription</h2>\n")
        for word in session.words:
            tooltip = f"{word.text} @ {word.start:.2f}s — conf: {word.confidence:.2f}"
            f.write(f"<span class='word {word.color}' title='{tooltip}'>{html_escape(word.text)}</span> ")

        # 📷 Screenshots
        if session.screenshots:
            f.write("<h2>Captures d’écran</h2>\n")
            for s in session.screenshots:
                f.write("<div class='screenshot'>\n")
                f.write(f"<p><strong>{s.filename}</strong> — {s.timestamp:.2f}s</p>\n")
                img_path = os.path.relpath(os.path.join("..", session.audio_path, "..", s.filename), output_folder)
                f.write(f"<img src='{img_path}' alt='{s.filename}' style='max-width: 400px;'>\n")
                f.write("</div>\n")

        # 💡 Insights
        if session.insights:
            f.write("<h2>Insights</h2>\n")
            for insight in session.insights:
                f.write("<div class='insight'>\n")
                f.write(f"<p><strong>@{insight.timestamp:.2f}s</strong><br>{html_escape(insight.text)}</p>\n")
                if insight.tags:
                    f.write(f"<p>🔖 Tags : {', '.join(insight.tags)}</p>\n")
                f.write(f"<p>📊 LoQ : {insight.loq} — Zoom: {insight.zoom_business}</p>\n")
                if insight.impact:
                    f.write(f"<p>🎯 Impact : {', '.join(insight.impact)}</p>\n")
                f.write("</div>\n")

        f.write("</body></html>")

    print(f"[REPORT] ✅ Rapport généré : {filepath}")
    return filepath
