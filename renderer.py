# -----------------------------
# File: renderer.py
# -----------------------------
import subprocess
import tempfile
import re
import os
import pandas as pd

def render_gloss(tokens, gloss_csv: str = "gloss_index.csv") -> str:
    """
    Given a list of gloss tokens, stitch video segments via ffmpeg concat
    and return path to output MP4.
    """
    # Load CSV mapping
    df = pd.read_csv(gloss_csv)
    index = {row['gloss']: row['path'] for _, row in df.iterrows()}

    # Normalize & collect file paths
    filelist = []
    for tok in tokens:
        key = re.sub(r"\d+|\(.*?\)", "", tok.upper()).strip()
        path = index.get(key)
        if not path:
            raise KeyError(f"No video for token '{tok}'")
        filelist.append(os.path.abspath(path))

    # Create temporary list file
    concat_txt = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
    for fpath in filelist:
        concat_txt.write(f"file '{fpath}'\n")
    concat_txt.flush()

    # Output MP4
    out_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_txt.name,
        "-c", "copy", out_mp4
    ]
    subprocess.run(cmd, check=True)
    return out_mp4