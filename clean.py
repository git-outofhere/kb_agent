#-- Remove the 2nd URL containing the word "print" which is exactly same as first URL except containing "default"
#-- Overwrite each text file w this update in clean_text folder. Don't concatenate any files 
#-- Note: Pre-processing files are a bit clunky in method so will need to mod to make it more elegant when time permits.
#-- Erika Figuero will update on new docs added/removed/modified

from pathlib import Path
import  re

# Directory containing .txt files
input_dir = Path("/home/shiraoka/projects/kb_agent/data/kbcontents/clean_text")

# Pattern to match URLs inside parentheses
paren_pattern = re.compile(r"\(\s*(https?://[^)]+)\s*\)")

# Process each text file (excluding all.txt and page_0.txt)
txt_files = sorted(
    f for f in input_dir.glob("*.txt")
    if f.name not in ("all.txt", "page_0.txt")
)

for txt_file in txt_files:
    content = txt_file.read_text(encoding="utf-8")

    # Find all URLs wrapped in parentheses
    paren_urls = paren_pattern.findall(content)

    # If there's a second URL that's the first with 'default'→'print', remove it (and its parentheses)
    if len(paren_urls) >= 2:
        first, second = paren_urls[0], paren_urls[1]
        if second == first.replace("default", "print"):
            # Strip out "(second_url)" including its parentheses
            content = re.sub(
                rf"\(\s*{re.escape(second)}\s*\)",
                "",
                content,
                count=1
            )

    # Overwrite the original file with the cleaned content
    txt_file.write_text(content, encoding="utf-8")

print(f"✅ Processed {len(txt_files)} files — removed second 'print' URLs where detected.")
