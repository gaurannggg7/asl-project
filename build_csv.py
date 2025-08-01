from pathlib import Path
import re,csv,sys

root = Path(sys.argv[1] if len(sys.argv)>1 else "asl_media")
rows=[]
for p in root.rglob("*"):
    if p.suffix.lower() not in {".mp4",".gif",".mkv",".webm"}: continue
    word=re.match(r"([A-Za-z]+)",p.stem).group(1).lower()
    date=re.search(r"(\d{4}-\d{1,2}-\d{1,2})",str(p))
    rows.append({
        "path":str(p),
        "word":word,
        "date":date.group(1) if date else "",
        "token":word.upper()
    })

with open("media_index.csv","w",newline="") as f:
    csv.DictWriter(f,rows[0].keys()).writeheader(); csv.DictWriter(f,rows[0].keys()).writerows(rows)
print(len(rows),"rows written to media_index.csv")
