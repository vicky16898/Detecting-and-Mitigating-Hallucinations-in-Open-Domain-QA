#!/usr/bin/env bash
# Download a single classifier checkpoint from Google Drive using gdown.
# Usage: bash scripts/fetch_checkpoints.sh <model_name> <strategy>
# Example: bash scripts/fetch_checkpoints.sh llama3base8b multi_layer_mean
#
# Files on Google Drive are named flat: <model>_<strategy>.pt
# Fill in the shareable file IDs below after uploading.

set -euo pipefail

MODEL=${1:?"Usage: $0 <model_name> <strategy>"}
STRATEGY=${2:?"Usage: $0 <model_name> <strategy>"}

declare -A GDRIVE_URLS
# ── Paste full shareable URLs here (Drive filename: <model>_<strategy>.pt) ────
GDRIVE_URLS["llama3base8b|original"]="https://drive.google.com/file/d/1HtTyK1UtFghiydmymJq8NL_KCQwfeY3D/view?usp=sharing"
GDRIVE_URLS["llama3base8b|multi_layer"]="https://drive.google.com/file/d/15AoRN7qWbtfeO6n-9mnqV22tqYvpCfPf/view?usp=sharing"
GDRIVE_URLS["llama3base8b|multi_layer_last_token"]="https://drive.google.com/file/d/1K_n_twb50o_NSmt_CCz_F5Z8TeD1-_35/view?usp=sharing"
GDRIVE_URLS["llama3base8b|multi_layer_mean"]="https://drive.google.com/file/d/1D19ISE9Jy68ddPDLaZrq8RR760ouUcL-/view?usp=sharing"
GDRIVE_URLS["llama3base8b|multi_layer_deltas"]="https://drive.google.com/file/d/1UoCeleo-shgyVoHAcblGDwCHv3UfC-AU/view?usp=sharing"
GDRIVE_URLS["gptj7b|original"]="https://drive.google.com/file/d/1h-kUBK6Y3Wt8In6sy86kgT0ATB7NNjMZ/view?usp=sharing"
GDRIVE_URLS["gptj7b|multi_layer"]="https://drive.google.com/file/d/17TJChxOB3ivi_pzO0HvL0brqqAcKQS9d/view?usp=sharing"
GDRIVE_URLS["gptj7b|multi_layer_last_token"]="https://drive.google.com/file/d/18SlqztHcJkMoRpyPyXtnkXOzqVdaZnmE/view?usp=sharing"
GDRIVE_URLS["gptj7b|multi_layer_mean"]="https://drive.google.com/file/d/1sNSk6qJMc01r0lXnBeF9YJXafqvteY9I/view?usp=sharing"
GDRIVE_URLS["gptj7b|multi_layer_deltas"]="https://drive.google.com/file/d/1UlBzx2kQFWmLdDj3GVgHmzT_bVtl3Rqk/view?usp=sharing"
# ─────────────────────────────────────────────────────────────────────────────

KEY="${MODEL}|${STRATEGY}"
URL="${GDRIVE_URLS[$KEY]:-}"

if [[ -z "$URL" || "$URL" == "GDRIVE_URL_PLACEHOLDER" ]]; then
    echo "No Google Drive URL registered for model='${MODEL}' strategy='${STRATEGY}'."
    echo "Edit scripts/fetch_checkpoints.sh and fill in the GDRIVE_URLS entry."
    exit 1
fi

DEST="data/auto-labeled/output/${MODEL}/${STRATEGY}/train_log/best_acc_model.pt"
mkdir -p "$(dirname "$DEST")"

echo "Downloading ${MODEL}_${STRATEGY}.pt from Google Drive ..."
gdown --fuzzy "$URL" -O "$DEST"
echo "Saved to ${DEST}"
