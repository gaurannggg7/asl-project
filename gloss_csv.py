#!/usr/bin/env python3
"""
Script to scan the 'media/SG ASL Dictionary' folder (or a user-specified directory)
and generate a CSV mapping each gloss token to its video file path.
The output CSV will have two columns:
  - gloss: uppercase ASL word (including "(Alt)" when present, but without numbers)
  - path: relative or absolute path to the video file

This version:
 - Recurses all subdirectories
 - Determines the token as all filename parts between 'SG ASL' and the first purely numeric or date-like segment
 - Keeps parentheses content (e.g., Alt) intact
 - Defaults to 'media/SG ASL Dictionary' if no input-dir is provided and that folder exists.
"""
import os
import re
import argparse
import pandas as pd


def normalize_filename(filename: str) -> str:
    """
    Extract the gloss token from filenames like:
    'SG ASL HELLO 0123 20240101 CC.mp4' -> 'HELLO'
    'SG ASL CAT(Alt) 0456 20230905 CC.mp4' -> 'CAT(Alt)'
    """
    parts = filename.rsplit('/', 1)[-1].split()
    # Must start with SG ASL
    if len(parts) < 4 or parts[0].upper() != 'SG' or parts[1].upper() != 'ASL':
        return None
    token_parts = []
    # Collect parts until we hit a numeric/date or known suffix like 'Upload' or 'CC'
    for part in parts[2:]:
        # pure digits or 8-digit date
        if re.fullmatch(r"\d+", part) or re.fullmatch(r"\d{8}", part):
            break
        # 'Upload' or 'CC' prefix signals end
        if part.upper().startswith('UPLOAD') or part.upper().startswith('CC'):
            break
        token_parts.append(part)
    if not token_parts:
        return None
    # join and uppercase
    token = ' '.join(token_parts).upper()
    # remove stray punctuation except parentheses
    token = re.sub(r"[^A-Z0-9()]+", '', token)
    return token


def build_gloss_index(base_dir: str) -> pd.DataFrame:
    records = []
    for root, _, files in os.walk(base_dir):
        for fn in files:
            gloss = normalize_filename(fn)
            if gloss:
                path = os.path.join(root, fn)
                records.append({'gloss': gloss, 'path': path})
    df = pd.DataFrame(records)
    df.drop_duplicates(subset=['gloss'], keep='first', inplace=True)
    return df


def main():
    default_dir = 'media/SG ASL Dictionary'
    # if default exists, use it; else require -i
    parser = argparse.ArgumentParser(
        description='Generate gloss_index.csv for SG ASL Dictionary videos'
    )
    parser.add_argument(
        '-i', '--input-dir',
        default=default_dir if os.path.isdir(default_dir) else None,
        help='Path to SG ASL Dictionary folder (will recurse)'
    )
    parser.add_argument(
        '-o', '--output-csv',
        default='gloss_index.csv',
        help='Output CSV filename'
    )
    args = parser.parse_args()

    if not args.input_dir or not os.path.isdir(args.input_dir):
        parser.error(f"Input directory not found: {args.input_dir}")

    df = build_gloss_index(args.input_dir)
    df.to_csv(args.output_csv, index=False)
    print(f"Generated '{args.output_csv}' with {len(df)} gloss entries.")


if __name__ == '__main__':
    main()
