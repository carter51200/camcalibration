"""
Stereo Rectification Quality Check (full script)

What it does:
1) Extract chessboard corners from original (images/left,right) and rectified (rectified/left,right).
2) Save corners to CSV-like TXT (corner_index,x,y).
3) Compute Δy = y_left - y_right before/after rectification.
4) Write per-image stats and overall stats to CSV.
5) Make plots (histogram/boxplot) and side-by-side visualizations with epipolar lines.

Notes:
- Comments and printed messages are in English.
- Rectified files may have the suffix "_rectified" (e.g., lm_L_1_rectified.png).
- Supports common image extensions and optional zero-padding (1, 01, 001).
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List

# =========================
# USER CONFIG
# =========================
# Input folders
LEFT_DIR = "images/left"
RIGHT_DIR = "images/right"
LEFT_RECT_DIR = "rectified/left"
RIGHT_RECT_DIR = "rectified/right"

# Output folders
OUT_COORDS_BEFORE = "out_coords/before"
OUT_COORDS_AFTER = "out_coords/after"
OUT_STATS_DIR = "out_stats"
OUT_VIZ_DIR = "out_viz"  # for side-by-side rectified overlays

# Chessboard inner corner size (columns, rows)
CHESSBOARD_SIZE = (9, 6)

# Image base name prefixes
LEFT_PREFIX = "lm_L_"
RIGHT_PREFIX = "lm_R_"

# Indices to process (1..31)
INDICES = list(range(1, 32))

# Visualization
ENABLE_PLOTS = True  # hist/boxplot PNGs
DRAW_GRID_STEP = 40  # grid spacing in pixels for overlay

# Finder options
RECT_SUFFIX = "_rectified"
PAD_WIDTHS = [0, 2, 3]  # try 1, 01, 001
VALID_EXTS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]


# =========================
# UTILS
# =========================
def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _candidates(dir_path: str, base: str, idx: int, extra_suffix: str = ""):
    """Yield candidate full paths like dir/base{idx}{extra}.ext with zero-pads."""
    for w in PAD_WIDTHS:
        idx_str = f"{idx:0{w}d}" if w > 0 else str(idx)
        for ext in VALID_EXTS:
            yield os.path.join(dir_path, f"{base}{idx_str}{extra_suffix}.{ext}")


def find_image_smart(dir_path: str, base: str, idx: int, allow_rectified: bool = False) -> Optional[str]:
    """
    Try multiple name variants:
      - base{idx}.[ext], base{idx:02d}.[ext], base{idx:03d}.[ext]
      - base{idx}_rectified.[ext] (if allow_rectified)
      - glob fallback
    """
    # 1) exact candidates
    for p in _candidates(dir_path, base, idx, ""):
        if os.path.isfile(p):
            return p

    # 2) rectified candidates (only if requested)
    if allow_rectified:
        for p in _candidates(dir_path, base, idx, RECT_SUFFIX):
            if os.path.isfile(p):
                return p

    # 3) glob fallback (handles odd cases)
    pats = [f"{base}{idx}*", f"{base}{idx:02d}*", f"{base}{idx:03d}*"]
    for pat in pats:
        g = sorted(glob.glob(os.path.join(dir_path, pat)))
        for p in g:
            if os.path.splitext(p)[1].lower().lstrip(".") in VALID_EXTS:
                return p

    # 4) not found -> helpful debug
    tried = list(_candidates(dir_path, base, idx, ""))
    if allow_rectified:
        tried += list(_candidates(dir_path, base, idx, RECT_SUFFIX))
    print(f"[DEBUG] Not found for index {idx} in '{dir_path}'. Tried patterns like:")
    print("        " + "\n        ".join(tried[:6] + (["..."] if len(tried) > 6 else [])))
    return None


def find_chessboard_corners(image: np.ndarray, chessboard_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Find 2D pixels of chessboard corners with subpixel refinement. Returns (N,1,2) or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if not found:
        return None
    corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners_subpix


def save_corners_txt(filepath: str, corners: np.ndarray) -> None:
    """Save corners as CSV-like text with header: corner_index,x,y"""
    corners_reshaped = corners.reshape(-1, 2)
    indices = np.arange(len(corners_reshaped)).reshape(-1, 1)
    data_to_save = np.hstack((indices, corners_reshaped))
    np.savetxt(
        filepath,
        data_to_save,
        fmt=['%d', '%.6f', '%.6f'],
        delimiter=',',
        header='corner_index,x,y',
        comments=''
    )


def stats_of_delta(delta_y: np.ndarray) -> dict:
    """Return key stats for Δy vector."""
    if delta_y.size == 0:
        return {
            "count": 0, "mean": np.nan, "std": np.nan, "median": np.nan,
            "mean_abs": np.nan, "max_abs": np.nan, "p95_abs": np.nan
        }
    abs_vals = np.abs(delta_y)
    return {
        "count": int(delta_y.size),
        "mean": float(np.mean(delta_y)),
        "std": float(np.std(delta_y, ddof=1)) if delta_y.size > 1 else 0.0,
        "median": float(np.median(delta_y)),
        "mean_abs": float(np.mean(abs_vals)),
        "max_abs": float(np.max(abs_vals)),
        "p95_abs": float(np.percentile(abs_vals, 95)),
    }


def write_stats_csv(path: str, header: List[str], rows: List[List]) -> None:
    """Write a simple CSV."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def draw_grid(img, step=40):
    """Draw faint horizontal grid lines to visually judge alignment."""
    out = img.copy()
    h, w = out.shape[:2]
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w-1, y), (200, 200, 200), 1, cv2.LINE_AA)
    return out


def epipolar_overlay(left_img, right_img, corners_left, corners_right, save_path, step_filter=5):
    L = draw_grid(left_img, step=DRAW_GRID_STEP)
    R = draw_grid(right_img, step=DRAW_GRID_STEP)

    h = max(L.shape[0], R.shape[0])
    w = L.shape[1] + R.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:L.shape[0], :L.shape[1]] = L
    canvas[:R.shape[0], L.shape[1]:L.shape[1]+R.shape[1]] = R

    ptsL = corners_left.reshape(-1, 2)
    ptsR = corners_right.reshape(-1, 2)

    for idx, ((xl, yl), (xr, yr)) in enumerate(zip(ptsL, ptsR)):
        if idx % step_filter != 0:
            continue  # only every Nth corner

        dy = yl - yr
        err = abs(dy)

        # Color based on error magnitude
        if err < 0.5:
            col = (0, 255, 0)  # green
        elif err < 2.0:
            col = (0, 255, 255)  # yellow
        else:
            col = (0, 0, 255)  # red

        # Left point
        xl_int, yl_int = int(round(xl)), int(round(yl))
        cv2.circle(canvas, (xl_int, yl_int), 3, col, -1, cv2.LINE_AA)

        # Right point (shifted by left width)
        xr_int = int(round(L.shape[1] + xr))
        yr_int = int(round(yr))
        cv2.circle(canvas, (xr_int, yr_int), 3, col, -1, cv2.LINE_AA)

        # Connect the two points directly
        cv2.line(canvas, (xl_int, yl_int), (xr_int, yr_int), col, 1, cv2.LINE_AA)

        # Δy text near the middle of the line
        mid_x = (xl_int + xr_int) // 2
        mid_y = (yl_int + yr_int) // 2
        cv2.putText(canvas, f"{dy:.1f}", (mid_x+5, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, canvas)




def process_pair(
    left_path: str, right_path: str,
    left_rect_path: str, right_rect_path: str,
    idx: int
):
    """Process one index: extract corners, compute Δy before/after, save coords & return stats."""
    # Load images
    L = cv2.imread(left_path)
    R = cv2.imread(right_path)
    Lr = cv2.imread(left_rect_path)
    Rr = cv2.imread(right_rect_path)

    if any(img is None for img in [L, R, Lr, Rr]):
        return None, f"Image load failed (index {idx})"

    # Detect corners
    cL = find_chessboard_corners(L, CHESSBOARD_SIZE)
    cR = find_chessboard_corners(R, CHESSBOARD_SIZE)
    cLr = find_chessboard_corners(Lr, CHESSBOARD_SIZE)
    cRr = find_chessboard_corners(Rr, CHESSBOARD_SIZE)

    # Validate detections
    if cL is None or cR is None:
        return None, f"Chessboard not found (original) index {idx}"
    if cLr is None or cRr is None:
        return None, f"Chessboard not found (rectified) index {idx}"

    # Ensure same number of corners
    n = cL.shape[0]
    if not (cR.shape[0] == n and cLr.shape[0] == n and cRr.shape[0] == n):
        return None, f"Corner count mismatch index {idx}"

    # Save corner coordinates
    save_corners_txt(os.path.join(OUT_COORDS_BEFORE, f"{LEFT_PREFIX}{idx}.txt"), cL)
    save_corners_txt(os.path.join(OUT_COORDS_BEFORE, f"{RIGHT_PREFIX}{idx}.txt"), cR)
    save_corners_txt(os.path.join(OUT_COORDS_AFTER, f"{LEFT_PREFIX}{idx}_rectified.txt"), cLr)
    save_corners_txt(os.path.join(OUT_COORDS_AFTER, f"{RIGHT_PREFIX}{idx}_rectified.txt"), cRr)

    # Compute Δy before & after
    yL = cL[:, 0, 1]
    yR = cR[:, 0, 1]
    yLr = cLr[:, 0, 1]
    yRr = cRr[:, 0, 1]

    dy_before = yL - yR
    dy_after = yLr - yRr

    # Stats
    stats_before = stats_of_delta(dy_before)
    stats_after = stats_of_delta(dy_after)

    return {
        "idx": idx,
        "dy_before": dy_before,
        "dy_after": dy_after,
        "stats_before": stats_before,
        "stats_after": stats_after,
        # before corners/images
        "cL": cL, "cR": cR,
        "L_img": L, "R_img": R,
        # after corners/images
        "cLr": cLr, "cRr": cRr,
        "Lr_img": Lr, "Rr_img": Rr
    }, None


def main():
    ensure_dirs([OUT_COORDS_BEFORE, OUT_COORDS_AFTER, OUT_STATS_DIR, OUT_VIZ_DIR])

    per_image_rows = []
    all_dy_before = []
    all_dy_after = []
    failures = []

    print(f"Processing {len(INDICES)} pairs...")

    for i in tqdm(INDICES, desc="Processing"):
        Lp  = find_image_smart(LEFT_DIR,       LEFT_PREFIX,  i, allow_rectified=False)
        Rp  = find_image_smart(RIGHT_DIR,      RIGHT_PREFIX, i, allow_rectified=False)
        Lrp = find_image_smart(LEFT_RECT_DIR,  LEFT_PREFIX,  i, allow_rectified=True)   # allow _rectified
        Rrp = find_image_smart(RIGHT_RECT_DIR, RIGHT_PREFIX, i, allow_rectified=True)   # allow _rectified

        if not all([Lp, Rp, Lrp, Rrp]):
            failures.append(f"Missing files index {i} (left/right/rectified)")
            continue

        result, err = process_pair(Lp, Rp, Lrp, Rrp, i)
        if err:
            failures.append(err)
            continue

        # === Before/After 시각화 저장 ===
        viz_before_path = os.path.join(OUT_VIZ_DIR, f"pair_{i:02d}_before.jpg")
        epipolar_overlay(result["L_img"], result["R_img"], result["cL"], result["cR"], viz_before_path)
        viz_after_path = os.path.join(OUT_VIZ_DIR, f"pair_{i:02d}_after.jpg")
        epipolar_overlay(result["Lr_img"], result["Rr_img"], result["cLr"], result["cRr"], viz_after_path)

    # ===============================
        # Collect stats
        sb = result["stats_before"]
        sa = result["stats_after"]

        per_image_rows.append([
            i,
            sb["count"], f"{sb['mean']:.6f}", f"{sb['std']:.6f}", f"{sb['median']:.6f}",
            f"{sb['mean_abs']:.6f}", f"{sb['max_abs']:.6f}", f"{sb['p95_abs']:.6f}",
            sa["count"], f"{sa['mean']:.6f}", f"{sa['std']:.6f}", f"{sa['median']:.6f}",
            f"{sa['mean_abs']:.6f}", f"{sa['max_abs']:.6f}", f"{sa['p95_abs']:.6f}",
        ])

        all_dy_before.append(result["dy_before"])
        all_dy_after.append(result["dy_after"])

        # Save rectified overlay visualization
        viz_path = os.path.join(OUT_VIZ_DIR, f"rectified_pair_{i:02d}.jpg")
        epipolar_overlay(result["Lr_img"], result["Rr_img"], result["cLr"], result["cRr"], viz_path)

    # Aggregate
    all_dy_before = np.concatenate(all_dy_before) if len(all_dy_before) else np.array([])
    all_dy_after = np.concatenate(all_dy_after) if len(all_dy_after) else np.array([])

    overall_before = stats_of_delta(all_dy_before)
    overall_after = stats_of_delta(all_dy_after)

    # Write per-image CSV
    header = [
        "index",
        "count_before", "mean_before", "std_before", "median_before",
        "mean_abs_before", "max_abs_before", "p95_abs_before",
        "count_after", "mean_after", "std_after", "median_after",
        "mean_abs_after", "max_abs_after", "p95_abs_after",
    ]
    write_stats_csv(os.path.join(OUT_STATS_DIR, "per_image_stats.csv"), header, per_image_rows)

    # Write overall CSV
    overall_header = ["set", "count", "mean", "std", "median", "mean_abs", "max_abs", "p95_abs"]
    overall_rows = [
        [
            "before",
            overall_before["count"], f"{overall_before['mean']:.6f}", f"{overall_before['std']:.6f}",
            f"{overall_before['median']:.6f}", f"{overall_before['mean_abs']:.6f}",
            f"{overall_before['max_abs']:.6f}", f"{overall_before['p95_abs']:.6f}",
        ],
        [
            "after",
            overall_after["count"], f"{overall_after['mean']:.6f}", f"{overall_after['std']:.6f}",
            f"{overall_after['median']:.6f}", f"{overall_after['mean_abs']:.6f}",
            f"{overall_after['max_abs']:.6f}", f"{overall_after['p95_abs']:.6f}",
        ],
    ]
    write_stats_csv(os.path.join(OUT_STATS_DIR, "overall_stats.csv"), overall_header, overall_rows)

    # (Optional) plots
    if ENABLE_PLOTS:
        try:
            import matplotlib.pyplot as plt

            if overall_before["count"] > 0:
                plt.figure()
                plt.hist(all_dy_before, bins=50)
                plt.title("Δy before rectification")
                plt.xlabel("Δy (pixels)")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_STATS_DIR, "hist_dy_before.png"), dpi=150)
                plt.close()

            if overall_after["count"] > 0:
                plt.figure()
                plt.hist(all_dy_after, bins=50)
                plt.title("Δy after rectification")
                plt.xlabel("Δy (pixels)")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_STATS_DIR, "hist_dy_after.png"), dpi=150)
                plt.close()

            if overall_before["count"] > 0 and overall_after["count"] > 0:
                plt.figure()
                plt.boxplot([all_dy_before, all_dy_after], labels=["before", "after"])
                plt.title("Δy distribution (before vs after)")
                plt.ylabel("Δy (pixels)")
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_STATS_DIR, "boxplot_dy_before_after.png"), dpi=150)
                plt.close()
        except Exception as e:
            print(f"[WARN] Matplotlib plots skipped due to error: {e}")

    # Summary print
    print("\n================ SUMMARY ================")
    print(f"Processed pairs: {len(per_image_rows)} / {len(INDICES)}")
    print(f"[Before] count={overall_before['count']}, mean={overall_before['mean']:.4f}, "
          f"std={overall_before['std']:.4f}, mean|Δy|={overall_before['mean_abs']:.4f}, "
          f"p95|Δy|={overall_before['p95_abs']:.4f}")
    print(f"[After ] count={overall_after['count']}, mean={overall_after['mean']:.4f}, "
          f"std={overall_after['std']:.4f}, mean|Δy|={overall_after['mean_abs']:.4f}, "
          f"p95|Δy|={overall_after['p95_abs']:.4f}")

    if failures:
        print("\nFailures / Skipped:")
        for msg in failures:
            print(" -", msg)
    print("=========================================\n")
    print(f"📁 Corner CSVs: '{OUT_COORDS_BEFORE}' (before), '{OUT_COORDS_AFTER}' (after)")
    print(f"📈 Stats & plots: '{OUT_STATS_DIR}'")
    print(f"🖼️ Rectified overlays: '{OUT_VIZ_DIR}'")


if __name__ == "__main__":
    main()
