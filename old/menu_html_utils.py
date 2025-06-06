# menu_html_utils.py
import os

def generate_menu_tree_html(menu_tree, output_path):
    html = """<html><head><meta charset='utf-8'><title>Menu Tree</title></head><body><h1>Menu Tree</h1><ul>"""

    def recurse(node):
        label = node.get("label", "Sans nom")
        image = node.get("image")
        sub_html = f"<li><strong>{label}</strong>"
        if image:
            sub_html += f"<br><img src='screenshots/{image}' width='300' style='margin:5px 0;'><br>"
        children = node.get("children", [])
        if children:
            sub_html += "<ul>"
            for child in children:
                sub_html += recurse(child)
            sub_html += "</ul>"
        sub_html += "</li>"
        return sub_html

    for top_node in menu_tree:
        html += recurse(top_node)

    html += "</ul></body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[HTML] ✅ Fichier HTML généré : {output_path}")
