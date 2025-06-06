#session_data.py
# session_data.py

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import os

@dataclass
class Word:
    text: str
    start: float
    confidence: float
    color: str = "black"

@dataclass
class Screenshot:
    filename: str
    timestamp: float
    linked_word_index: Optional[int] = None
    comment: Optional[str] = None

@dataclass
class Insight:
    timestamp: float
    text: str
    tags: List[str] = field(default_factory=list)
    loq: int = 0
    zoom_business: str = "direct"
    impact: List[str] = field(default_factory=list)
    status: str = "draft"

@dataclass
class SessionData:
    session_id: str
    audio_path: str
    screenshots: List[Screenshot] = field(default_factory=list)
    words: List[Word] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    path: Optional[str] = field(default=None, repr=False, compare=False)  # ← ajouté ici

    def save(self, output_folder: str):
        self.path = output_folder  # ← on stocke le chemin ici
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, f"{self.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        print(f"[SAVE] ✅ Session sauvegardée dans {path}")


    @staticmethod
    def load(path: str) -> "SessionData":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        session = SessionData(
            session_id=data["session_id"],
            audio_path=data["audio_path"],
            screenshots=[Screenshot(**s) for s in data.get("screenshots", [])],
            words=[Word(**w) for w in data.get("words", [])],
            insights=[Insight(**i) for i in data.get("insights", [])]
        )
        session.path = os.path.dirname(path)  # ← on retrouve le dossier de session
        return session


    def export_session_view(self):
        from session_view_generator import generate_session_view
        return generate_session_view(self)
