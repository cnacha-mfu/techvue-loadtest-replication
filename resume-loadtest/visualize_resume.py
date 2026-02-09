#!/usr/bin/env python3
"""
Analyze Resume Load-Test CSVs and generate visuals + markdown report.

Expected detailed CSV columns (from resume_load_test_* generator):
- request_id, applicant_id, file_name, bytes
- create_status, create_ms
- upload_status, upload_ms
- extract_status, extract_ms
- analyze_status, analyze_ms
- photo_url, ok, stage, error
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import numpy as np

# --------------------
# Logging
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("resume-report")

# --------------------
# File discovery
# --------------------
def find_latest_resume_files():
    detailed_files = glob.glob("resume_load_test_detailed_*.csv")
    if not detailed_files:
        raise FileNotFoundError("No resume detailed test files found (resume_load_test_detailed_*.csv)")
    summary_files = glob.glob("resume_load_test_summary_*.csv")
    if not summary_files:
        logger.warning("No resume summary files found (resume_load_test_summary_*.csv) — continuing anyway.")
    detailed_files.sort(reverse=True)
    summary_files.sort(reverse=True)
    return detailed_files, summary_files

# --------------------
# Loading & prep
# --------------------
def _extract_concurrency_from_name(path: str) -> int:
    # resume_load_test_detailed_<concurrency>_<timestamp>.csv
    #                    idx:            3
    # split by '_' -> ["resume","signed","stress","detailed","<conc>","<timestamp>.csv"]
    parts = os.path.basename(path).split("_")
    try:
        # handle possible hyphens/extra underscores defensively
        # pattern guarantees concurrency at index 4 for "resume_load_test_detailed_*"
        return int(parts[4])
    except Exception:
        # fallback for slight naming drift
        for token in parts:
            if token.isdigit():
                return int(token)
        raise ValueError(f"Cannot parse concurrency from filename: {path}")

def load_resume_data(detailed_files):
    frames = []
    for f in detailed_files:
        conc = _extract_concurrency_from_name(f)
        df = pd.read_csv(f)
        df["Concurrency"] = conc
        # ensure numeric
        for col in ["create_ms", "upload_ms", "extract_ms", "analyze_ms",
                    "create_status", "upload_status", "extract_status", "analyze_status"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# --------------------
# Visualizations
# --------------------
def _stage_success_rate(df: pd.DataFrame, stage_status_col: str) -> pd.DataFrame:
    if stage_status_col not in df.columns:
        return pd.DataFrame(columns=["Concurrency", "Success Rate (%)"])
    res = df.groupby("Concurrency").apply(
        lambda x: (x[stage_status_col] == 200).mean() * 100
    ).reset_index()
    res.columns = ["Concurrency", "Success Rate (%)"]
    return res

def _time_percentiles(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        return pd.DataFrame()
    return df.groupby("Concurrency")[time_col].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).unstack()

def _box_violin_scatter(df: pd.DataFrame, time_col: str, status_col: str, stage_name: str, viz_dir: str):
    if time_col not in df.columns:
        return
    plt.figure(figsize=(12,6))
    sns.boxplot(x="Concurrency", y=time_col, data=df)
    plt.title(f"{stage_name} Time Distribution (Box)")
    plt.xlabel("Concurrency")
    plt.ylabel("Time (ms)")
    plt.savefig(f"{viz_dir}/{stage_name.lower()}_time_box.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12,6))
    sns.violinplot(x="Concurrency", y=time_col, data=df)
    plt.title(f"{stage_name} Time Distribution (Violin)")
    plt.xlabel("Concurrency")
    plt.ylabel("Time (ms)")
    plt.savefig(f"{viz_dir}/{stage_name.lower()}_time_violin.png", dpi=300, bbox_inches="tight")
    plt.close()

    if status_col in df.columns:
        plt.figure(figsize=(12,6))
        colors = (df[status_col] == 200)
        plt.scatter(df["Concurrency"], df[time_col], c=colors, alpha=0.7)
        plt.title(f"{stage_name} Times by Concurrency & Status")
        plt.xlabel("Concurrency")
        plt.ylabel("Time (ms)")
        plt.colorbar(label="Success (True=200)")
        plt.savefig(f"{viz_dir}/{stage_name.lower()}_time_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()

    avg = df.groupby("Concurrency")[time_col].agg(["mean","std"]).reset_index()
    plt.figure(figsize=(12,6))
    plt.errorbar(avg["Concurrency"], avg["mean"], yerr=avg["std"], fmt="o-", capsize=5, linewidth=2, markersize=8)
    plt.title(f"Average {stage_name} Time by Concurrency")
    plt.xlabel("Concurrency")
    plt.ylabel("Time (ms)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{viz_dir}/{stage_name.lower()}_time_avg.png", dpi=300, bbox_inches="tight")
    plt.close()

    pct = _time_percentiles(df, time_col)
    if not pct.empty:
        plt.figure(figsize=(12,6))
        pct.plot(marker="o", linewidth=2, markersize=6)
        plt.title(f"{stage_name} Time Percentiles by Concurrency")
        plt.xlabel("Concurrency")
        plt.ylabel("Time (ms)")
        plt.legend(title="Percentile", bbox_to_anchor=(1.05,1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(f"{viz_dir}/{stage_name.lower()}_time_percentiles.png", dpi=300, bbox_inches="tight")
        plt.close()

def _bar_success(df: pd.DataFrame, status_col: str, title: str, out_path: str):
    sr = _stage_success_rate(df, status_col)
    if sr.empty: 
        return
    plt.figure(figsize=(12,6))
    bars = plt.bar(sr["Concurrency"], sr["Success Rate (%)"])
    plt.title(title)
    plt.xlabel("Concurrency")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0,100)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x()+b.get_width()/2., h+1, f"{h:.1f}%", ha="center", va="bottom")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def create_visualizations(df: pd.DataFrame, viz_dir: str):
    os.makedirs(viz_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    stages = [
        ("create_ms",  "create_status",  "Create API"),
        ("upload_ms",  "upload_status",  "Upload"),
        ("extract_ms", "extract_status", "Extract Photo"),
        ("analyze_ms", "analyze_status", "Analyze"),
    ]
    for time_col, status_col, stage_name in stages:
        _box_violin_scatter(df, time_col, status_col, stage_name, viz_dir)
        _bar_success(df, status_col, f"{stage_name} Success Rate by Concurrency", 
                     f"{viz_dir}/{stage_name.lower().replace(' ','_')}_success_rate.png")

# --------------------
# Markdown report
# --------------------
def _stage_stats_table(df: pd.DataFrame, time_col: str, status_col: str, stage_name: str) -> str:
    cols = []
    if time_col in df.columns:
        tbl = df.groupby("Concurrency")[time_col].agg(["mean","std","min","max"]).round(2)
        tbl.columns = [f"Avg {stage_name} (ms)", f"Std (ms)", f"Min (ms)", f"Max (ms)"]
        cols.append(tbl)
    if status_col in df.columns:
        sr = _stage_success_rate(df, status_col).set_index("Concurrency")
        sr.columns = [f"{stage_name} Success Rate (%)"]
        cols.append(sr)
    if not cols:
        return ""
    merged = pd.concat(cols, axis=1)
    return merged.reset_index().to_markdown(index=False)

def generate_md_report(viz_dir: str, df: pd.DataFrame, timestamp: str):
    stages = [
        ("create_ms",  "create_status",  "Create API"),
        ("upload_ms",  "upload_status",  "Upload"),
        ("extract_ms", "extract_status", "Extract Photo"),
        ("analyze_ms", "analyze_status", "Analyze"),
    ]
    sections = []
    for time_col, status_col, stage in stages:
        table_md = _stage_stats_table(df, time_col, status_col, stage)
        img_block = f"""
### {stage} Visuals
![{stage} Box]({os.path.join(viz_dir, f"{stage.lower().replace(' ','_')}_time_box.png")})
![{stage} Violin]({os.path.join(viz_dir, f"{stage.lower().replace(' ','_')}_time_violin.png")})
![{stage} Scatter]({os.path.join(viz_dir, f"{stage.lower().replace(' ','_')}_time_scatter.png")})
![{stage} Avg]({os.path.join(viz_dir, f"{stage.lower().replace(' ','_')}_time_avg.png")})
![{stage} Percentiles]({os.path.join(viz_dir, f"{stage.lower().replace(' ','_')}_time_percentiles.png")})
![{stage} Success Rate]({os.path.join(viz_dir, f"{stage.lower().replace(' ','_')}_success_rate.png")})
        """.strip()
        sections.append(f"## {stage}\n\n{table_md}\n\n{img_block}\n")

    overall_sr = (df.get("analyze_status", pd.Series(dtype=float)) == 200).mean() * 100 if "analyze_status" in df.columns else np.nan
    joined_sections = "\n".join(sections)
    concurrency_levels = ", ".join(map(str, sorted(df["Concurrency"].unique())))
    success_rate = 0 if np.isnan(overall_sr) else f"{overall_sr:.1f}%"

    md = f"""# Resume Load Test Results - {timestamp}

    - **Total Requests:** {len(df)}
    - **Concurrency Levels Tested:** {concurrency_levels}
    - **Overall Analyze Success Rate:** {success_rate}

    {joined_sections}
    """

    with open(os.path.join(viz_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)

# --------------------
# Main
# --------------------
def main():
    try:
        detailed_files, summary_files = find_latest_resume_files()
        # logger.info("Found %d detailed and %d summary resume files", len(detailed_files), len(summary_files))
        df = load_resume_data(detailed_files)
        # logger.info("Loaded %,d rows across %d concurrency levels", len(df), df["Concurrency"].nunique())

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = f"resume_load_test_visualizations_{ts}"
        create_visualizations(df, viz_dir)
        generate_md_report(viz_dir, df, ts)

        print("\n" + "="*60)
        print("RESUME LOAD TEST SUMMARY")
        print("="*60)
        print(f"Total requests: {len(df)}")
        print(f"Concurrency levels: {sorted(df['Concurrency'].unique())}")
        if "analyze_status" in df.columns:
            print(f"Overall analyze success rate: {(df['analyze_status']==200).mean()*100:.1f}%")
        print(f"Report: {os.path.join(viz_dir, 'report.md')}")
        print("="*60 + "\n")

    except Exception as e:
        logger.exception("Error building resume report")
        raise

if __name__ == "__main__":
    main()