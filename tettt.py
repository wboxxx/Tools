import tkinter as tk
from tkinter import filedialog
import difflib
import os
import re
import json
# Optional prompt to clear config
CONFIG_FILE = "diff_config.json"

if os.path.exists(CONFIG_FILE):
    print(f"[CONFIG] Un fichier de config existe déjà : {CONFIG_FILE}")
    response = input("🟠 Voulez-vous le supprimer et re-sélectionner les fichiers ? (Y/n) ").strip().lower()
    if response == "y":
        try:
            os.remove(CONFIG_FILE)
            print("[CONFIG] Fichier de config supprimé.\n")
        except Exception as e:
            print(f"[CONFIG] ⚠️ Erreur lors de la suppression : {e}\n")
    else:
        print("[CONFIG] → Config conservée.\n")


def extract_functions(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    functions = []
    current_func = []
    in_function = False

    for line in lines:
        if line.strip().startswith("def ") or line.strip().startswith("class "):
            if in_function:
                functions.append(current_func)
                current_func = []
            in_function = True
        if in_function:
            current_func.append(line)

    if current_func:
        functions.append(current_func)

    return functions

def extract_function_names(func_blocks):
    names = []
    for block in func_blocks:
        for line in block:
            match = re.search(r"^(def|class) (\w+)", line.strip())
            if match:
                names.append((match.group(2), block))
                break
    return names

def get_function_dict(file_path):
    blocks = extract_functions(file_path)
    return dict(extract_function_names(blocks))

def compare_functions(original_funcs, modified_funcs):
    modified = []
    added = []
    removed = []

    for name, new_body in modified_funcs.items():
        if name in original_funcs:
            old_body = original_funcs[name]
            if old_body != new_body:
                diff_size = sum(1 for l1, l2 in zip(old_body, new_body) if l1 != l2)
                diff_size += abs(len(old_body) - len(new_body))
                modified.append((name, new_body, diff_size))
        else:
            added.append((name, new_body))

    for name, old_body in original_funcs.items():
        if name not in modified_funcs:
            removed.append((name, old_body))

    return (
        sorted(modified, key=lambda x: x[2], reverse=True),
        sorted(added, key=lambda x: len(x[1]), reverse=True),
        sorted(removed, key=lambda x: len(x[1]), reverse=True),
    )

def find_related_functions(keyword, func_dict):
    related = []
    for name, body in func_dict.items():
        if keyword in "".join(body):
            related.append((name, body))
    return related

def keyword_propagation_analysis(keyword, func_dict):
    propagation_hits = []
    for name, body in func_dict.items():
        joined = "".join(body)
        if re.search(rf"\b{keyword}\b", joined):
            propagation_hits.append((name, body))
            continue
        if any(re.search(rf"\b{keyword}\w*", line) for line in body):
            propagation_hits.append((name, body))
    return propagation_hits

def select_file(title):
    path = filedialog.askopenfilename(title=title, filetypes=[("Python Files", "*.py")])
    print(f"[FILE SELECT] {title} ➜ {path}")
    return path

def load_or_select_files():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            try:
                config = json.load(f)
                original_file = config.get("original_file")
                modified_file = config.get("modified_file")
                if os.path.exists(original_file) and os.path.exists(modified_file):
                    print("[CONFIG] Fichiers chargés depuis la config.")
                    return original_file, modified_file
            except json.JSONDecodeError:
                pass

    print("[CONFIG] Aucune config valide trouvée. Sélection manuelle requise.")
    original_file = select_file("Sélectionner la version ORIGINALE")
    modified_file = select_file("Sélectionner la version MODIFIÉE")

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"original_file": original_file, "modified_file": modified_file}, f)

    return original_file, modified_file


# === Lancement ===
print("[INIT] Lancement de la comparaison entre deux versions d’un fichier Python...")
root = tk.Tk()
root.withdraw()

original_file, modified_file = load_or_select_files()

print(f"[PARSE] Extraction des fonctions depuis : {original_file}")
original_funcs = get_function_dict(original_file)
print(f"[PARSE] {len(original_funcs)} fonctions extraites")

print(f"[PARSE] Extraction des fonctions depuis : {modified_file}")
modified_funcs = get_function_dict(modified_file)
print(f"[PARSE] {len(modified_funcs)} fonctions extraites")

print("[DIFF] Calcul des différences entre les fonctions...")
modified, added, removed = compare_functions(original_funcs, modified_funcs)
print(f"[DIFF] {len(modified)} fonctions modifiées, {len(added)} ajoutées, {len(removed)} supprimées")

keyword = input("[SEARCH] Mot-clé pour recherche récursive (laisser vide pour ignorer) : ").strip()
related = []
propagation = []

if keyword:
    print(f"[SEARCH] Recherche directe sur le mot-clé : {keyword}")
    related = find_related_functions(keyword, modified_funcs)
    print(f"[SEARCH] {len(related)} fonctions liées trouvées")

    print(f"[TRACE] Analyse de propagation depuis le mot-clé : {keyword}")
    propagation = keyword_propagation_analysis(keyword, modified_funcs)
    print(f"[TRACE] {len(propagation)} fonctions affectées par propagation détectées")

# === EXPORT RESULTATS ===
print("\n========== Résultats ==========")
with open("modified_functions_summary.py", "w", encoding="utf-8") as f:
    if keyword:
        if related:
            f.write("# ========== RELATED FUNCTIONS (direct keyword match) ==========\n\n")
            for name, body in related:
                f.write(f"# [RELATED] {name} (keyword: {keyword})\n")
                f.writelines(body)
                f.write("\n\n")
        if propagation:
            f.write("# ========== PROPAGATED FUNCTIONS (keyword propagation) ==========\n\n")
            for name, body in propagation:
                f.write(f"# [PROPAGATED] {name} (via keyword: {keyword})\n")
                f.writelines(body)
                f.write("\n\n")
    else:
        if modified:
            f.write("# ========== MODIFIED FUNCTIONS (diff size descending) ==========\n\n")
            for name, new_body, diff_size in modified:
                f.write(f"\ndef __diff__{name}():\n")
                f.write(f"    # [MODIFIED] {name} — diff size: {diff_size}\n")
                old_body = original_funcs.get(name, [])
                diff_lines = difflib.unified_diff(
                    old_body,
                    new_body,
                    fromfile='original',
                    tofile='modified',
                    lineterm='',
                    n=3
                )
                lines = list(diff_lines)
                if lines:
                    for line in lines:
                        f.write("    # " + line + "\n")
                else:
                    f.write("    # Aucun diff détecté\n")

                # 💡 Ligne factice non commentée pour permettre le repli Notepad++
                f.write("    return  # end of diff block\n\n")

        if added:
            f.write("# ========== ADDED FUNCTIONS ==========\n\n")
            for name, body in added:
                f.write(f"# [ADDED] {name}\n")
                f.writelines(body)
                f.write("\n\n")

        if removed:
            f.write("# ========== REMOVED FUNCTIONS ==========\n\n")
            for name, body in removed:
                f.write(f"# [REMOVED] {name}\n")
                f.writelines(body)
                f.write("\n\n")

print("\n✅ Résultats exportés dans modified_functions_summary.py")
# === OUVERTURE DANS NOTEPAD++ SI DISPONIBLE ===
notepadpp_path = r"C:\Program Files\Notepad++\notepad++.exe"
summary_file = os.path.abspath("modified_functions_summary.py")

if os.path.exists(notepadpp_path):
    print(f"[OPEN] Ouverture du fichier dans Notepad++...")
    import subprocess

    try:
        subprocess.Popen([notepadpp_path, summary_file])
        print(f"[OPEN] Fichier ouvert dans Notepad++ ✅")
    except Exception as e:
        print(f"[ERROR] Impossible d’ouvrir Notepad++ : {e}")
else:
    print(f"[OPEN] Notepad++ non trouvé à l’emplacement standard : {notepadpp_path}")
    print(f"[OPEN] Tu peux ouvrir manuellement ce fichier : {summary_file}")

