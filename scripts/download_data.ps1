# Download CTU-13, IoT-23 (lighter), and MedBIoT bulk pcaps into data/raw/.
# Mirrors scripts/download_data.sh — uses curl -C - so re-runs resume partial files.
#
# Usage:  pwsh -File scripts/download_data.ps1 [-Target ctu13|iot23|medbiot|all]
param(
    [ValidateSet("ctu13", "iot23", "medbiot", "all")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Dest = "data/raw"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$CTU13Url = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2"
$IOT23Url = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz"
$MedbiotBase = "https://cs.taltech.ee/research/data/medbiot/bulk/raw_dataset"
$MedbiotFiles = @(
    "malware/bashlite_mal_CC_all.pcap",
    "malware/bashlite_mal_spread_all.pcap",
    "malware/mirai_mal_CC_all.pcap",
    "malware/mirai_mal_spread_all.pcap",
    "malware/torii_mal_all.pcap",
    "normal/bashlite_leg.pcap",
    "normal/mirai_leg.pcap",
    "normal/torii_leg.pcap"
)

function Download-Ctu13 {
    $out = Join-Path $Dest "CTU-13-Dataset.tar.bz2"
    Write-Host ">> CTU-13 (~1.9 GB) -> $out"
    curl.exe -L -C - --retry 5 --retry-delay 10 -o $out $CTU13Url
    if ($LASTEXITCODE -ne 0) { throw "curl failed with $LASTEXITCODE" }
    Write-Host ">> Extracting CTU-13"
    tar.exe -xjf $out -C $Dest
    Write-Host ">> CTU-13 done."
}

function Download-Iot23 {
    $out = Join-Path $Dest "iot_23_datasets_small.tar.gz"
    Write-Host ">> IoT-23 lighter (~8.7 GB) -> $out"
    curl.exe -L -C - --retry 5 --retry-delay 10 -o $out $IOT23Url
    if ($LASTEXITCODE -ne 0) { throw "curl failed with $LASTEXITCODE" }
    Write-Host ">> Extracting IoT-23 lighter"
    tar.exe -xzf $out -C $Dest
    $scenarios = Join-Path $Dest "opt/Malware-Project/BigDataset/IoTScenarios"
    $link = Join-Path $Dest "IoT-23"
    if ((Test-Path $scenarios) -and -not (Test-Path $link)) {
        New-Item -ItemType Junction -Path $link -Target ((Resolve-Path $scenarios).Path) | Out-Null
        Write-Host ">> Created junction $link -> IoTScenarios"
    }
    Write-Host ">> IoT-23 done."
}

function Download-Medbiot {
    $root = Join-Path $Dest "medbiot/bulk/raw_dataset"
    New-Item -ItemType Directory -Force -Path (Join-Path $root "malware") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $root "normal") | Out-Null
    Write-Host ">> MedBIoT bulk raw pcaps (~10 GB total)"
    foreach ($rel in $MedbiotFiles) {
        $url = "$MedbiotBase/$rel"
        $out = Join-Path $root ($rel -replace "/", [IO.Path]::DirectorySeparatorChar)
        $dir = Split-Path $out -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        Write-Host ">> $rel"
        curl.exe -L -C - --retry 5 --retry-delay 10 -o $out $url
        if ($LASTEXITCODE -ne 0) { throw "curl failed for $rel ($LASTEXITCODE)" }
    }
    Write-Host ">> MedBIoT done."
}

switch ($Target) {
    "ctu13"   { Download-Ctu13 }
    "iot23"   { Download-Iot23 }
    "medbiot" { Download-Medbiot }
    "all"     { Download-Ctu13; Download-Iot23; Download-Medbiot }
}
