#menu_tree_builder.py



import re
from session_data import Screenshot
from collections import deque
from utils import Brint  # ou adapte selon ton import
def match_images_to_tags_by_interval(tags, screenshots):
    tag_ts = [(i, t["start"], t["label"]) for i, t in enumerate(tags) if t.get("type") == "MENU"]
    tag_ts.sort(key=lambda x: x[1])
    screenshots = sorted(screenshots, key=lambda s: s.timestamp)

    matches = {}
    for idx, (i, t_start, label) in enumerate(tag_ts):
        t_end = tag_ts[idx + 1][1] if idx + 1 < len(tag_ts) else float('inf')
        Brint(f"[MATCH] Tag '{label}' (#{i}) → Intervalle [{t_start:.2f}s, {t_end:.2f}s)")
        found = False
        for s in screenshots:
            if t_start <= s.timestamp < t_end:
                Brint(f"        ✅ Match: {s.filename} @ {s.timestamp:.2f}s")
                matches[i] = s
                found = True
                break
            else:
                Brint(f"        ❌ {s.filename} @ {s.timestamp:.2f}s (hors intervalle)")
        if not found:
            Brint(f"        ⚠️ Aucun screenshot matché pour '{label}'")
    return matches
def print_menu_tree(tree, indent=0):
    for node in tree:
        label = node.get("label", "???")
        ts = node.get("timestamp", "?")
        img = node.get("image", None)
        Brint("  " * indent + f"📁 {label} @ {ts}s" + (f" 🖼️ {img}" if img else ""))
        if node.get("children"):
            print_menu_tree(node["children"], indent + 1)


def build_menu_tree_from_tagged_text(tagged_text_lines, word_timeline, screenshots=None, parsed_tags=None):
    if screenshots is None:
        screenshots = []

    Brint("[MENU TREE] 🧠 Démarrage analyse des tags")
    tree = []
    current_parent = None
    node_stack = []
    last_timestamp = None

    # 🧠 Préparation des tags structurés avec timestamp
    parsed_tags = []
    for i, line in enumerate(tagged_text_lines):
        words = line.strip().split()
        if len(words) < 3 or words[1] != "MENU":
            Brint(f"[MENU TREE] ❌ Ligne ignorée (pas un tag MENU valide) : {line}")
            continue
        direction = words[2]
        label = " ".join(words[3:]) if len(words) > 3 else f"Menu_{i}"

        timestamp = None
        for w in word_timeline:
            w_text = w["word"] if isinstance(w, dict) else w.text
            w_start = w["start"] if isinstance(w, dict) else w.start
            if w_text == label or label.startswith(w_text):
                timestamp = w_start
                break
                
        if timestamp is None:
            timestamp = last_timestamp + 1.5 if last_timestamp else 0
        last_timestamp = timestamp

        parsed_tags.append({
            "type": "MENU",
            "direction": direction,
            "label": label,
            "start": timestamp,
            # "image": match_images_to_tags_by_interval(start_time, screenshots)  # ← ICI
        })

    # 📸 Matching screenshots une seule fois
    if parsed_tags is None:
        # fallback : reconstruire à partir des lignes taggées
        parsed_tags = []
        for i, line in enumerate(tagged_text_lines):
            parts = line.strip().split()
            if len(parts) >= 4 and parts[1] == "MENU":
                parsed_tags.append({
                    "type": "MENU",
                    "direction": parts[2],
                    "label": " ".join(parts[3:]),
                    "start": next((w["start"] for w in word_timeline if w["word"] == parts[3] or parts[3].startswith(w["word"])), 0)
                })

    matched_images = match_images_to_tags_by_interval(parsed_tags, screenshots)

    # 🌲 Construction de l'arbre
    for idx, tag in enumerate(parsed_tags):
    # for tag in parsed_tags:
        direction = tag["direction"]
        label = tag["label"]
        timestamp = tag["start"]

        Brint(f"[MENU TREE] ➕ Tag détecté : direction={direction}, label='{label}'")

        matched = matched_images.get(idx)
        # matched = matched_images.get(label)
        if matched:
            Brint(f"[MENU TREE] 🖼️ Image matchée pour {label} → {matched.filename}")
            img = matched.filename
        else:
            Brint(f"[MENU TREE] ⚠️ Aucune image trouvée pour {label}")
            img = None

        node = {
            "label": label,
            "timestamp": timestamp,
            "image": img,
            "children": [],
            "depth": len(node_stack)
        }

        if direction == "ROOT":
            tree.append(node)
            node_stack = [node]
            Brint(f"[MENU TREE] 🌲 Nouveau ROOT : {label}")
        elif direction == "DOWN":
            if node_stack:
                node_stack[-1]["children"].append(node)
                node_stack.append(node)
                Brint(f"[MENU TREE]   ⬇️ Ajout DOWN sous {node_stack[-2]['label']}")
            else:
                tree.append(node)
                Brint(f"[MENU TREE] ⚠️ DOWN sans ROOT, ajouté au niveau racine")
        elif direction == "SIDE":
            if len(node_stack) > 1:
                node_stack[-2]["children"].append(node)
                node_stack[-1] = node
                Brint(f"[MENU TREE]   ⬅️ Ajout SIDE au même niveau que {label}")
            else:
                tree.append(node)
                Brint(f"[MENU TREE] ⚠️ SIDE sans contexte, ajouté au niveau racine")
        elif direction == "BACK":
            if len(node_stack) > 1:
                popped = node_stack.pop()
                Brint(f"[MENU TREE] 🔙 BACK : retour de {popped['label']} à {node_stack[-1]['label']}")
            else:
                Brint(f"[MENU TREE] ⚠️ BACK sans stack, reste au niveau racine")
            continue  # ← on ne crée pas de node pour BACK


        elif direction == "UP":
            if len(node_stack) > 1:
                node_stack.pop()  # remonte d’un cran
                node_stack[-1]["children"].append(node)
                node_stack.append(node)  # ← on descend d’un cran avec le nouveau node
                Brint(f"[MENU TREE]   🔼 UP : retour à {node_stack[-2]['label']}")
            else:
                tree.append(node)
                Brint(f"[MENU TREE] ⚠️ UP sans stack, ajouté au niveau racine")

        else:
            tree.append(node)
            Brint(f"[MENU TREE] ⚠️ Direction inconnue '{direction}', ajout racine")

    Brint("[MENU] ✅ Arborescence finale construite.")
    print_menu_tree(tree)
    return tree
