"""Run the entire pipeline: prepare data -> EDA -> train & evaluate.

    python run_pipeline.py            # real LIAR data (download first)
    python run_pipeline.py --demo     # synthetic dry-run data (NOT real!)
"""
import os
import sys

from src import eda, prepare_data, train

if __name__ == "__main__":
    if "--demo" in sys.argv:
        from src import make_demo_data
        make_demo_data.main()
        os.environ["DATASET_NOTE"] = "SYNTHETIC DEMO DATA (not real)"
    else:
        from src import download_data
        if not download_data.already_present():
            if download_data.main() != 0:
                sys.exit(1)
    prepare_data.main(os.environ.get("DATASET_NOTE", "LIAR (PolitiFact)"))
    eda.main()
    train.main()
    print("\nAll done. Launch the GUI with:  streamlit run app.py")
