#!/usr/bin/env python3
"""Merge coconut_large + coconut_xlarge into a single JSON index.

Design
------
- Each record gets a fresh sequential integer `merged_id` starting at 0.
- Large records are emitted first (in their original annotation order) so their
  ids occupy [0, N_large). Xlarge records continue from N_large upward.
- For every record we record:
      merged_id         : new sequential id
      source            : "large" | "xlarge"
      object365_name    : e.g. "objects365_v2_02241588"  (stem, no extension)
      panseg_file       : e.g. "objects365_v2_02241588.png"  (basename only)
      panseg_subdir     : directory under the source root where the png lives
                          - large : "panoptic_object365"
                          - xlarge: "coconuts_xlarge/panseg"
      height, width     : best-effort, may be null for xlarge (no image record)
      segments_info     : list of {id, category_id, isthing, area} (normalized)
      original_image_file (optional) : original obj365 jpg filename if known
- De-dup: xlarge records whose object365_name already appeared in large are
  dropped (large-priority). 189 such duplicates are expected.
- Reading the panseg png from a record:
      os.path.join(SOURCE_ROOT[rec["source"]], rec["panseg_subdir"], rec["panseg_file"])
  where SOURCE_ROOT is configured at load-time. The merged JSON itself stays
  portable (no absolute paths baked in).

Output: one JSON file (default: merged_coconut.json) with structure:
  {
    "info": {...summary...},
    "categories": [... copied from large's JSON ...],
    "source_roots": {                # hint for consumers (absolute paths at build time)
        "large":  "...",
        "xlarge": "..."
    },
    "records": [ ... ]
  }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------------
# Defaults: point at the HF cache snapshot paths used in this workspace. Can be
# overridden on the CLI.
# ----------------------------------------------------------------------------
DEFAULT_LARGE_ROOT = (
    "/root/.cache/huggingface/hub/datasets--xdeng77--coconut_large/"
    "snapshots/ce8605783330498d380d2ef2e52fcbe5194419b3"
)
DEFAULT_XLARGE_ROOT = (
    "/root/.cache/huggingface/hub/datasets--xdeng77--coconut_xlarge/"
    "snapshots/2f9c829748bc52b186b4ae180a9917fcdd7840ab/coconuts_xlarge"
)

LARGE_PANSEG_SUBDIR = "panoptic_object365"
LARGE_JSON_REL = "panseg_object365_train_v2.json"

XLARGE_PANSEG_SUBDIR = "panseg"
XLARGE_PANSEG_INFO_SUBDIR = "panseg_info"


OBJ365_RE = re.compile(r"(objects365_v2_\d+)")


# ---------------------------- helpers ----------------------------
def _normalize_segments(seg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the canonical keys and coerce types. Missing keys -> None/0."""
    out = []
    for s in seg_list or []:
        try:
            area = float(s["area"]) if "area" in s and s["area"] is not None else None
        except (TypeError, ValueError):
            area = None
        out.append({
            "id": int(s["id"]) if "id" in s else None,
            "category_id": int(s["category_id"]) if "category_id" in s else None,
            "isthing": bool(s.get("isthing", False)),
            "area": area,
        })
    return out


def _obj_name_from(x: str) -> Optional[str]:
    if not x:
        return None
    m = OBJ365_RE.search(x)
    return m.group(1) if m else None


# ---------------------------- large loader ----------------------------
def load_large(large_root: str) -> List[Dict[str, Any]]:
    """Produce a list of normalized records from coconut_large.

    The upstream JSON has two heterogeneous annotation shapes; we resolve both
    to a uniform record that points at `panoptic_object365/<obj_name>.png`.
    """
    json_path = os.path.join(large_root, LARGE_JSON_REL)
    panseg_dir = os.path.join(large_root, LARGE_PANSEG_SUBDIR)
    print(f"[large] loading {json_path} ...", flush=True)
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # index images by two possible keys
    img_by_id: Dict[int, Dict[str, Any]] = {}
    img_by_file: Dict[str, Dict[str, Any]] = {}
    for im in d.get("images", []):
        if "id" in im:
            img_by_id[im["id"]] = im
        if "file_name" in im:
            img_by_file[im["file_name"]] = im

    existing_png = set(os.listdir(panseg_dir))

    records: List[Dict[str, Any]] = []
    missing = 0
    for ann in d["annotations"]:
        # Resolve object365 name across both shapes.
        obj_name = ann.get("object365_file_name")
        if not obj_name:
            # shape A: annotation.file_name is already "<obj>.png"
            obj_name = _obj_name_from(ann.get("file_name", ""))
        if not obj_name:
            # last resort: via image record keyed by image_id
            im = img_by_id.get(ann.get("image_id"))
            if im:
                obj_name = _obj_name_from(
                    im.get("object365_file_name") or im.get("object365_name") or "")
        if not obj_name:
            missing += 1
            continue

        png_file = f"{obj_name}.png"
        if png_file not in existing_png:
            missing += 1
            continue

        # height/width via image record if possible
        h = w = None
        orig_file = None
        im = None
        fn = ann.get("file_name")
        if fn and fn in img_by_file:
            im = img_by_file[fn]
        elif ann.get("image_id") in img_by_id:
            im = img_by_id[ann["image_id"]]
        if im:
            h = im.get("height")
            w = im.get("width")
            orig_file = im.get("file_name")

        records.append({
            "source": "large",
            "object365_name": obj_name,
            "panseg_file": png_file,
            "panseg_subdir": LARGE_PANSEG_SUBDIR,
            "height": h,
            "width": w,
            "segments_info": _normalize_segments(ann.get("segments_info", [])),
            "original_image_file": orig_file,
        })

    print(f"[large] produced {len(records)} records, "
          f"{missing} skipped (unresolved png)", flush=True)
    # Return categories as well
    return records, d.get("categories", [])


# ---------------------------- xlarge loader ----------------------------
def load_xlarge(xlarge_root: str) -> List[Dict[str, Any]]:
    info_dir = os.path.join(xlarge_root, XLARGE_PANSEG_INFO_SUBDIR)
    panseg_dir = os.path.join(xlarge_root, XLARGE_PANSEG_SUBDIR)
    print(f"[xlarge] scanning {info_dir} ...", flush=True)
    info_files = [f for f in os.listdir(info_dir) if f.endswith(".json")]
    info_files.sort()
    existing_png = set(os.listdir(panseg_dir))
    print(f"[xlarge] {len(info_files)} info files, {len(existing_png)} pngs",
          flush=True)

    records: List[Dict[str, Any]] = []
    missing = 0
    t0 = time.time()
    for i, fn in enumerate(info_files):
        obj_name = os.path.splitext(fn)[0]
        png_file = obj_name + ".png"
        if png_file not in existing_png:
            missing += 1
            continue
        with open(os.path.join(info_dir, fn), "r", encoding="utf-8") as f:
            seg = json.load(f)
        records.append({
            "source": "xlarge",
            "object365_name": obj_name,
            "panseg_file": png_file,
            "panseg_subdir": f"{XLARGE_PANSEG_SUBDIR}",   # relative to xlarge root
            "height": None,
            "width": None,
            "segments_info": _normalize_segments(seg),
            "original_image_file": None,
        })
        if (i + 1) % 20000 == 0:
            rate = (i + 1) / max(1e-9, time.time() - t0)
            print(f"[xlarge] {i+1}/{len(info_files)} "
                  f"({rate:.0f} files/s)", flush=True)

    print(f"[xlarge] produced {len(records)} records, "
          f"{missing} skipped", flush=True)
    return records


# ---------------------------- merge ----------------------------
def merge(large_root: str, xlarge_root: str, out_path: str) -> None:
    large_records, categories = load_large(large_root)
    xlarge_records = load_xlarge(xlarge_root)

    seen: Dict[str, int] = {}       # object365_name -> merged_id
    merged: List[Dict[str, Any]] = []

    def _push(rec: Dict[str, Any]) -> None:
        on = rec["object365_name"]
        if on in seen:
            return        # large-priority de-dup
        mid = len(merged)
        rec["merged_id"] = mid
        seen[on] = mid
        merged.append(rec)

    for r in large_records:
        _push(r)
    n_after_large = len(merged)

    x_dup = 0
    for r in xlarge_records:
        if r["object365_name"] in seen:
            x_dup += 1
            continue
        _push(r)

    info = {
        "description": "Merged index of coconut_large + coconut_xlarge, "
                       "sequentially re-numbered (large first).",
        "large_count": n_after_large,
        "xlarge_new_count": len(merged) - n_after_large,
        "xlarge_dropped_duplicates": x_dup,
        "total": len(merged),
    }
    print(f"[merge] {info}", flush=True)

    out = {
        "info": info,
        "categories": categories,
        "source_roots": {
            "large":  os.path.abspath(large_root),
            "xlarge": os.path.abspath(xlarge_root),
        },
        "records": merged,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[merge] wrote {out_path} ({size_mb:.1f} MB)", flush=True)


# ---------------------------- reader helper ----------------------------
def panseg_path_for(record: Dict[str, Any],
                    source_roots: Dict[str, str]) -> str:
    """Return the absolute png path for a merged record.

    Example:
        with open(merged_path) as f: doc = json.load(f)
        roots = doc["source_roots"]
        for rec in doc["records"][:5]:
            print(rec["merged_id"], panseg_path_for(rec, roots))
    """
    root = source_roots[record["source"]]
    return os.path.join(root, record["panseg_subdir"], record["panseg_file"])


# ---------------------------- cli ----------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--large_root", default=DEFAULT_LARGE_ROOT,
                   help="coconut_large snapshot directory")
    p.add_argument("--xlarge_root", default=DEFAULT_XLARGE_ROOT,
                   help="coconut_xlarge snapshot directory (the one that "
                        "contains panseg/ and panseg_info/)")
    p.add_argument("-o", "--output", default="merged_coconut.json",
                   help="Path to write the merged JSON")
    p.add_argument("--verify", type=int, default=5,
                   help="After merging, verify the first N png paths exist")
    args = p.parse_args()

    merge(args.large_root, args.xlarge_root, args.output)

    # quick verification
    if args.verify > 0:
        with open(args.output, "r", encoding="utf-8") as f:
            doc = json.load(f)
        roots = doc["source_roots"]
        for rec in doc["records"][:args.verify]:
            p = panseg_path_for(rec, roots)
            print(f"  merged_id={rec['merged_id']:>7d}  source={rec['source']:6s}  "
                  f"exists={os.path.isfile(p)}  {p}")
        # also verify last xlarge record
        for rec in reversed(doc["records"]):
            if rec["source"] == "xlarge":
                p = panseg_path_for(rec, roots)
                print(f"  merged_id={rec['merged_id']:>7d}  source=xlarge  "
                      f"exists={os.path.isfile(p)}  {p}")
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
