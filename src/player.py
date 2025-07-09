# src/player.py
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips

# ── change this if you moved the folder elsewhere ────────────────────────────
_ASL_DIR = Path(__file__).parent.parent / "asl_videos"
# ----------------------------------------------------------------------------

def _find_clip(gloss: str) -> Path | None:
    """
    Return the first video file that exists for *gloss* regardless of extension.
    """
    for ext in (".mp4", ".webm", ".mkv"):       # add more if you need them
        fp = _ASL_DIR / f"{gloss.lower()}{ext}"
        if fp.exists():
            return fp
    return None


def play_glosses(glosses: list[str]) -> None:
    """
    Given a list of ASL gloss strings (e.g. ["hello", "borrow"]),
    locate matching video files, concatenate them, and preview the result.
    """
    clips = []
    for g in glosses:
        clip_path = _find_clip(g)
        if clip_path:
            clips.append(VideoFileClip(clip_path.as_posix()))
        else:
            print(f"[WARN] no clip found for: {g}")

    if not clips:
        print("[ERROR] Nothing to play — no matching video files.")
        return

    final = concatenate_videoclips(clips, method="compose")
    final = final.without_audio()
    final.preview()                 # opens a window and plays immediately
