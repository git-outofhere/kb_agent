#-- RUN THIS AFTER BUILDING CLEAN_TEXT FILES:
#-- THIS WILL RENAME THE FILES TO SOMETHING MORE SUGGESTIVE OF THE CONTENTS

import os
import re
import shutil

# ───────────────────────────────────────────────
#  CONFIGURATION
# ───────────────────────────────────────────────
INPUT_DIR  = "/home/shiraoka/projects/kb_agent/data/kbcontents/clean_text"
OUTPUT_DIR = "/home/shiraoka/projects/kb_agent/data/kbcontents/renamed_clean_text"

# Words to skip when picking your "first N" or when trimming trailing words
STOPWORDS = {
    "a", "an", "the",
    "of", "in", "on", "at", "by", "for", "with",
    "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "over", "under"
}

# ───────────────────────────────────────────────
#  MAKE SURE OUTPUT DIR EXISTS
# ───────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────────────────────────────────────────────
#  PROCESS EACH FILE
# ───────────────────────────────────────────────
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(".txt"):
        continue

    in_path = os.path.join(INPUT_DIR, fname)
    with open(in_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1) find the first closing parenthesis of a URL, then grab the rest of that line
    m = re.search(r"\)\s*([^\r\n]+)", text)
    if m:
        title_line = m.group(1)
    else:
        # fallback: first non-blank line
        for line in text.splitlines():
            if line.strip():
                title_line = line
                break
        else:
            continue  # empty file

    # 2) split into words (keeping apostrophes)
    words = re.findall(r"\b\w+'\w+|\w+\b", title_line)

    # 3) build initial selection: prefer filtered words, else raw
    filtered = [w for w in words if w.lower() not in STOPWORDS]
    selected = filtered[:4] if filtered else words[:4]

    # 4) trim any trailing stopwords
    while selected and selected[-1].lower() in STOPWORDS:
        selected.pop()

    # 5) if we’ve stripped everything, fall back to the first non-stopword
    if not selected:
        first_nonstop = next((w for w in words if w.lower() not in STOPWORDS), None)
        if first_nonstop:
            selected = [first_nonstop]
        else:
            # give up on renaming; use original name
            new_fname = fname
            out_path  = os.path.join(OUTPUT_DIR, new_fname)
            shutil.move(in_path, out_path)
            print(f"{fname} → (no change, all words were stopwords)")
            continue

    # 6) assemble new filename and move
    new_fname = "_".join(selected) + ".txt"
    out_path  = os.path.join(OUTPUT_DIR, new_fname)
    shutil.move(in_path, out_path)
    print(f"{fname} → {new_fname}")
