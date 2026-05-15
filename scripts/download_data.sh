#!/usr/bin/env bash
# Download CTU-13 and IoT-23 (lighter) into data/raw/.
# MedBIoT must be downloaded manually — see data/README.md.
#
# Usage:  bash scripts/download_data.sh [ctu13|iot23|all]
#
# All curl invocations use -C - so re-running resumes partial downloads.

set -euo pipefail

cd "$(dirname "$0")/.."
DEST="data/raw"
mkdir -p "$DEST"

CTU13_URL="https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2"
IOT23_URL="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz"

download_ctu13() {
    local out="$DEST/CTU-13-Dataset.tar.bz2"
    echo ">> CTU-13 (~1.9 GB) → $out"
    curl -L -C - --retry 5 --retry-delay 10 -o "$out" "$CTU13_URL"

    echo ">> Extracting CTU-13"
    tar -xjf "$out" -C "$DEST"
    echo ">> CTU-13 done. Layout:"
    ls "$DEST/CTU-13-Dataset/" | head
}

download_iot23() {
    local out="$DEST/iot_23_datasets_small.tar.gz"
    echo ">> IoT-23 lighter (~8.7 GB) → $out"
    curl -L -C - --retry 5 --retry-delay 10 -o "$out" "$IOT23_URL"

    echo ">> Extracting IoT-23 lighter"
    tar -xzf "$out" -C "$DEST"
    echo ">> IoT-23 done. Layout:"
    ls "$DEST/" | grep -i iot | head
}

case "${1:-all}" in
    ctu13) download_ctu13 ;;
    iot23) download_iot23 ;;
    all)   download_ctu13; download_iot23 ;;
    *)     echo "usage: $0 [ctu13|iot23|all]"; exit 1 ;;
esac

echo
echo "MedBIoT is NOT auto-downloaded — see data/README.md for the bulk pcap URLs."
