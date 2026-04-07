#!/usr/bin/env python3
"""Merge partial JSON files from multi-GPU Step 1 into a single JSON.

Usage:
    python merge_partial_jsons.py --input_dir /path/to/input --output_json /path/to/new.json
"""

import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Merge partial JSON files into one.")
    parser.add_argument("--input_dir", required=True, help="Directory containing new_partial_*.json files")
    parser.add_argument("--output_json", required=True, help="Output merged JSON path")
    args = parser.parse_args()

    partials = []
    for pf in sorted(glob.glob(os.path.join(args.input_dir, "new_partial_*.json"))):
        with open(pf) as f:
            partials.extend(json.load(f))

    # Sort by video_path to ensure deterministic order
    partials.sort(key=lambda x: x["video_path"])

    with open(args.output_json, "w") as f:
        json.dump(partials, f, indent=4, ensure_ascii=False)

    # Clean up partial files
    for pf in glob.glob(os.path.join(args.input_dir, "new_partial_*.json")):
        os.remove(pf)

    print(f"Merged {len(partials)} entries into {args.output_json}")


if __name__ == "__main__":
    main()
