"""Run the course-project experiments with ONE command.

    python run_experiments.py --features ..\\features_local --out ..\\local_results
    python run_experiments.py --features ..\\features_local --out ..\\local_results --only 4 5 6
    python run_experiments.py --features ..\\features_local --baseline ..\\features_kaggle --out ..\\local_results --only 7
    python run_experiments.py --features ..\\features_local --out ..\\local_results --skip-neural
    python run_experiments.py --features ..\\features_local --out ..\\local_results --quick       (smoke test)
    python run_experiments.py ... --install-model ..\\system                                         (copy final_system.joblib there)

--features may be any folder that contains features.npz and meta.csv (also in a subfolder).
Each experiment runs in its own process, so one failure does not stop the others. A summary table is printed and saved.
"""
import argparse, os, sys, shutil, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = {   # id: (script, output sub-folder, files that must exist when it succeeded, needs PyTorch)
    1: ("02_exp1_baselines.py", "exp1", ["exp1_main_results.csv"], False),
    2: ("03_exp2_fusion.py", "exp2", ["exp2_summary_mean.csv"], True),
    3: ("04_exp3_generalization.py", "exp3", ["exp3_all_long.csv"], True),
    4: ("05_exp4_ablation.py", "exp4", ["exp4_A_retrain_mean.csv", "exp4_B_drop_one.csv"], False),
    5: ("06_exp5_error_analysis.py", "exp5", ["exp5_hard_cases.csv", "exp5_coverage_accuracy.csv"], False),
    6: ("07_exp6_explanations.py", "exp6", ["exp6_T1_localization.csv", "final_system.joblib"], False),
    7: ("08_exp7_frames.py", "exp7", ["exp7_summary.csv"], False),     # needs features from extract_local.py (per-frame embeddings)
}


def find_features(root):
    root = Path(root)
    hits = [p.parent for p in root.rglob("features.npz") if (p.parent / "meta.csv").exists()]
    if not hits:
        sys.exit(f"No folder containing both features.npz and meta.csv was found under {root}.\n"
                 f"If you only have parts\\*.npz files, the merge step of the extraction notebook is needed first.")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--baseline", help="folder with the Kaggle features.npz + meta.csv (used by Experiment 7 as the reference)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", type=int, nargs="*", choices=sorted(EXPERIMENTS), help="experiment numbers, e.g. --only 4 5 6")
    ap.add_argument("--skip-neural", action="store_true", help="skip Exp 2 and 3 (they use PyTorch)")
    ap.add_argument("--quick", action="store_true", help="smoke test: few epochs / one split where the scripts support it")
    ap.add_argument("--install-model", metavar="SYSTEM_DIR", help="copy exp6/final_system.joblib into this folder when done")
    a = ap.parse_args()

    feat = find_features(a.features)
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    chosen = a.only or sorted(EXPERIMENTS)
    if a.skip_neural:
        chosen = [e for e in chosen if not EXPERIMENTS[e][3]]
    print(f"features: {feat}\nresults : {out}\nrunning : experiments {chosen}\n")

    rows = []
    for e in chosen:
        script, sub, must, _ = EXPERIMENTS[e]
        d = out / sub; d.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, FEAT_DIR=str(feat), EXP_OUT=str(d), MPLBACKEND="Agg", PYTHONUTF8="1",
                   PYTHONIOENCODING="utf-8", QUICK="1" if a.quick else "0", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
        if a.baseline:
            env["BASE_DIR"] = str(find_features(a.baseline))
        print(f"=== Experiment {e}: {script} ===", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / script)], env=env, cwd=str(HERE))
        missing = [f for f in must if not (d / f).exists()]
        status = "OK" if r.returncode == 0 and not missing else ("FAILED" if r.returncode else "OUTPUT MISSING")
        rows.append((e, script, status, f"{time.time() - t0:.0f}s", ", ".join(missing)))
        print(f"--- Experiment {e}: {status}\n", flush=True)

    print("\nSUMMARY")
    print(f"{'exp':<4}{'script':<30}{'status':<16}{'time':<8}missing")
    for r in rows:
        print(f"{r[0]:<4}{r[1]:<30}{r[2]:<16}{r[3]:<8}{r[4]}")
    (out / "run_summary.csv").write_text("exp,script,status,time,missing\n" + "\n".join(",".join(map(str, r)) for r in rows), encoding="utf-8")

    if a.install_model:
        src = out / "exp6" / "final_system.joblib"
        if src.exists():
            shutil.copy(src, Path(a.install_model) / "final_system.joblib")
            print(f"\ncopied final_system.joblib -> {a.install_model}")
        else:
            print("\n--install-model: exp6/final_system.joblib not found (run Experiment 6 first)")
    sys.exit(0 if all(r[2] == "OK" for r in rows) else 1)


if __name__ == "__main__":
    main()
