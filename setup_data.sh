#!/bin/bash
# Downloads the data/ folder from Google Drive, removes the old one, and extracts the new one.
# Requires: gdown (pip install gdown)
set -e

GDRIVE_LINK="https://drive.google.com/file/d/1psbyqWfN9VmO7XhQLqoql_QU2lqaOkv0/view?usp=sharing"
ARCHIVE="data.zip"

echo "Downloading data archive from Google Drive..."
gdown --fuzzy "${GDRIVE_LINK}" -O "$ARCHIVE"

echo "Removing old data/ folder..."
rm -rf data/

echo "Extracting archive..."
unzip -q "$ARCHIVE"
rm "$ARCHIVE"

echo "Done. data/ is ready."
