import os
import sys
import shutil
import subprocess
import zipfile

def run_setup():
    gdrive_id = "1psbyqWfN9VmO7XhQLqoql_QU2lqaOkv0"
    archive = "data.zip"

    print("--- 1. Downloading data archive from Google Drive ---")
    try:
        # Pass the file ID directly (gdown 6.0+ removed --fuzzy)
        subprocess.run([sys.executable, "-m", "gdown", gdrive_id, "-O", archive], check=True)
    except Exception as e:
        print(f"Error downloading with gdown: {e}")
        print("Make sure you have gdown installed: pip install gdown")
        return

    print("\n--- 2. Removing old data/ folder ---")
    if os.path.exists("data"):
        shutil.rmtree("data")
        print("Old data/ folder removed.")

    print("\n--- 3. Extracting archive ---")
    try:
        # ZipFile is native to Python and works on all OSs
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return

    print("\n--- 4. Cleaning up ---")
    if os.path.exists(archive):
        os.remove(archive)
        print(f"Removed {archive}")

    print("\nDone. data/ is ready.")

if __name__ == "__main__":
    run_setup()
