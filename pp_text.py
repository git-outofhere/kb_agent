#-- Pre-process the text...removed tailor footers, headers
#-- code to remove the table of contents from each text file as well as the bottom footer

from pathlib import Path
import re

# Paths to your source and destination folders
input_dir = Path("/home/shiraoka/projects/kb_agent/data/kbcontents/text2")  #updated on 9/9/2025
#input_dir = Path("/home/shiraoka/projects/kb_agent/data/kbcontents/text")
output_dir = Path("/home/shiraoka/projects/kb_agent/data/kbcontents/clean_text")
output_dir.mkdir(parents=True, exist_ok=True)

# Markers for top and bottom removal
top_marker = "X-ray Services"
#bottom_marker = "Central California Alliance for Health"
bottom_marker = "Your Content Copyright Notice Knowledge Base Software"


for file_path in input_dir.glob("*.txt"):
    full_text = file_path.read_text(encoding="utf-8")

    # 1) Remove everything from start through top_marker and its following URL
    top_idx = full_text.find(top_marker)
    if top_idx != -1:
        tail = full_text[top_idx + len(top_marker):]
        # find first URL after the marker
        url_match = re.search(r"https?://\S+", tail)
        if url_match:
            start_after = url_match.end()
            middle = tail[start_after:]
        else:
            middle = tail
    else:
        middle = full_text

    # 2) Remove everything from bottom_marker onward
    bottom_idx = middle.find(bottom_marker)
    if bottom_idx != -1:
        cleaned = middle[:bottom_idx]
    else:
        cleaned = middle

    # 3) Deduplicate URLs in parentheses
    seen = set()
    def dedupe(match):
        url = match.group(1)
        if url in seen:
            return ""
        seen.add(url)
        return f"({url})"
    cleaned = re.sub(r"\((https?://[^)]+)\)", dedupe, cleaned)
    # remove any empty parentheses
    cleaned = re.sub(r"\(\)", "", cleaned)

    # Strip leading/trailing blank lines
    cleaned = cleaned.lstrip("\r\n").rstrip("\r\n")

    # Write out the cleaned text
    (output_dir / file_path.name).write_text(cleaned, encoding="utf-8")

print("✅ All files processed: removed content through 'X-ray Services' and its URL; truncated at footer; URLs de-duplicated.")
