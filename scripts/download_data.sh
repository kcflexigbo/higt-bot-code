#!/usr/bin/env bash
# Download CTU-13, IoT-23 (lighter), and MedBIoT bulk pcaps into data/raw/.
#
# Usage:  bash scripts/download_data.sh [ctu13|iot23|medbiot|all]
#
# All curl invocations use -C - so re-running resumes partial downloads.

set -euo pipefail

cd "$(dirname "$0")/.."
DEST="data/raw"
mkdir -p "$DEST"

CTU13_URL="https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2"
IOT23_URL="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz"
MEDBIOT_BASE="https://cs.taltech.ee/research/data/medbiot/bulk/raw_dataset"
MEDBIOT_FILES=(
    "malware/bashlite_mal_CC_all.pcap"        # 236 MB
    "malware/bashlite_mal_spread_all.pcap"    # 295 MB
    "malware/mirai_mal_CC_all.pcap"           # 666 MB
    "malware/mirai_mal_spread_all.pcap"       # 148 MB
    "malware/torii_mal_all.pcap"              #  25 MB
    "normal/bashlite_leg.pcap"                # 6.0 GB
    "normal/mirai_leg.pcap"                   # 1.66 GB
    "normal/torii_leg.pcap"                   # 1.07 GB
)

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

    # The tar packs everything under opt/Malware-Project/BigDataset/IoTScenarios/.
    # Symlink to a sane path so code can reference data/raw/IoT-23/...
    if [ -d "$DEST/opt/Malware-Project/BigDataset/IoTScenarios" ] && [ ! -e "$DEST/IoT-23" ]; then
        ln -sfn opt/Malware-Project/BigDataset/IoTScenarios "$DEST/IoT-23"
        echo ">> Created symlink $DEST/IoT-23"
    fi

    echo ">> IoT-23 done. Layout:"
    ls "$DEST/IoT-23/" 2>/dev/null | head
}

download_medbiot() {
    local root="$DEST/medbiot/bulk/raw_dataset"
    mkdir -p "$root/malware" "$root/normal"
    echo ">> MedBIoT bulk raw pcaps (~10 GB total: 5 malware + 3 normal)"
    for rel in "${MEDBIOT_FILES[@]}"; do
        local url="$MEDBIOT_BASE/$rel"
        local out="$root/$rel"
        echo ">> ($rel)"
        curl -L -C - --retry 5 --retry-delay 10 -o "$out" "$url"
    done
    echo ">> MedBIoT done. Layout:"
    ls -la "$root/malware" "$root/normal"
}

case "${1:-all}" in
    ctu13)   download_ctu13 ;;
    iot23)   download_iot23 ;;
    medbiot) download_medbiot ;;
    all)     download_ctu13; download_iot23; download_medbiot ;;
    *)       echo "usage: $0 [ctu13|iot23|medbiot|all]"; exit 1 ;;
esac
