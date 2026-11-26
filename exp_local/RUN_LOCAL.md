# Run all experiments locally with one command (works in cmd or PowerShell)

    cd C:\Users\Aryavart\NLP_Project\exp_local
    ..\.venv\Scripts\activate
    pip install pandas matplotlib
    python run_experiments.py --features ..\features_local --out ..\local_results --install-model ..\system

Options: --only 4 5 6   --skip-neural (skips Exp 2 and 3)   --quick (smoke test)
--features can be any folder that contains features.npz and meta.csv (subfolders are searched).
