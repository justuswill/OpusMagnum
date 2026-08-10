#!/usr/bin/env python3
"""
Fetch every puzzle tracked by zlbb.faendir.com — campaign, journal (XCIX & CVIII),
DRM, and community — plus top leaderboard solutions for the key metric categories.

For normal/polymer puzzles: GX (min cost>cycles·area), AX (min area>cost·cycles),
                            CX (min cycles>cost·area), SUM (min g+c+a),
                            GC (min cost, cycles as tiebreak)
For production puzzles:     GX_P, CX_P, SUM_P, IX_P (instructions)

Puzzle files come from the leaderboard's /om/puzzle/{id}/file endpoint.
Solutions come from /om/puzzle/{id}/records and follow the redirect to GitHub.

Folder structure: puzzles/{collection}/{id}-{SafeName}/
Existing files are skipped (safe to re-run).
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

BASE_URL = "https://zlbb.faendir.com"
PUZZLES_DIR = os.path.join(os.path.dirname(__file__), "puzzles")

# Categories to collect per puzzle type
NORMAL_CATS   = {"GX", "AX", "CX", "SUM", "GC"}
PRODUCTION_CATS = {"GX_P", "CX_P", "SUM_P", "IX_P"}
PRODUCTION_TYPES = {"PRODUCTION"}


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise


def fetch_binary(url, retries=3):
    """Download binary content, following redirects. Falls back to master branch
    if the pinned commit hash returns 404 from GitHub raw."""
    def _try_master_fallback(original_url):
        """When the leaderboard redirect lands on a 404 at a pinned commit,
        retry at master — the file may have been added after that commit."""
        if "raw.githubusercontent.com" in original_url and "/om-leaderboard/" in original_url:
            path = original_url.split("/om-leaderboard/")[1].split("/", 1)[1]
            fallback = f"https://raw.githubusercontent.com/f43nd1r/om-leaderboard/master/{path}"
            with urllib.request.urlopen(fallback, timeout=30) as resp:
                return resp.read()
        raise FileNotFoundError(f"No master fallback for {original_url}")

    last_url = url
    for attempt in range(retries):
        try:
            req = urllib.request.Request(last_url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                last_url = resp.url  # track redirected URL for fallback
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                try:
                    return _try_master_fallback(last_url)
                except Exception:
                    pass
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise


def safe_name(s):
    """Convert a display name to a filesystem-safe string."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", s).strip("-")


def collection_subdir(collection_id):
    return {
        "CAMPAIGN":      "campaign",
        "JOURNAL_XCIX":  "journal_xcix",
        "JOURNAL_CVIII": "journal_cviii",
        "DRM":           "drm",
        "COMMUNITY":     "community",
    }.get(collection_id, collection_id.lower())


def target_categories(puzzle_type):
    return PRODUCTION_CATS if puzzle_type in PRODUCTION_TYPES else NORMAL_CATS


def score_str(score):
    area = score.get("area")
    area_s = f"{area}a" if area is not None else "na"
    return f"{score['cost']}g-{score['cycles']}c-{area_s}-{score['instructions']}i"


def main():
    print("Fetching full puzzle list from leaderboard...")
    all_puzzles = fetch_json(f"{BASE_URL}/om/puzzles")
    print(f"Total puzzles: {len(all_puzzles)}")

    errors = []

    for puzzle in all_puzzles:
        pid        = puzzle["id"]
        name       = puzzle["displayName"]
        ptype      = puzzle["type"]
        coll_id    = puzzle["group"]["collection"]["id"]
        subdir     = collection_subdir(coll_id)
        folder_name = f"{pid}-{safe_name(name)}"
        folder_path = os.path.join(PUZZLES_DIR, subdir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        print(f"\n[{subdir}] {pid}: {name} ({ptype})")

        # ── Puzzle file ──────────────────────────────────────────────────────
        puzzle_dest = os.path.join(folder_path, f"{pid}.puzzle")
        if os.path.exists(puzzle_dest):
            print(f"  .puzzle exists, skipping")
        else:
            try:
                data = fetch_binary(f"{BASE_URL}/om/puzzle/{pid}/file")
                with open(puzzle_dest, "wb") as f:
                    f.write(data)
                print(f"  puzzle: {len(data)} bytes")
            except Exception as e:
                msg = f"  ERROR downloading puzzle {pid}: {e}"
                print(msg)
                errors.append(msg)

        # ── Records ──────────────────────────────────────────────────────────
        try:
            records = fetch_json(
                f"{BASE_URL}/om/puzzle/{pid}/records?includeFrontier=true"
            )
        except Exception as e:
            msg = f"  ERROR fetching records for {pid}: {e}"
            print(msg)
            errors.append(msg)
            continue

        target_cats = target_categories(ptype)
        cat_records = {}
        for record in records:
            for cat_id in record.get("categoryIds", []):
                if cat_id in target_cats and cat_id not in cat_records:
                    cat_records[cat_id] = record

        found = sorted(cat_records.keys())
        missing = sorted(target_cats - set(found))
        print(f"  categories found: {found}" + (f"  missing: {missing}" if missing else ""))

        for cat_id, record in cat_records.items():
            sol_url = record.get("solution")
            if not sol_url:
                continue
            sc = record["score"]
            fname = f"{pid}_{cat_id}_{score_str(sc)}.solution"
            dest  = os.path.join(folder_path, fname)
            if os.path.exists(dest):
                print(f"  {cat_id}: exists")
                continue
            try:
                data = fetch_binary(sol_url)
                with open(dest, "wb") as f:
                    f.write(data)
                print(f"  {cat_id}: {fname} ({len(data)} bytes)")
            except Exception as e:
                msg = f"  ERROR {cat_id} solution for {pid}: {e}"
                print(msg)
                errors.append(msg)
            time.sleep(0.05)

    print("\n" + "="*60)
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
    else:
        print("All done — no errors.")


if __name__ == "__main__":
    main()
