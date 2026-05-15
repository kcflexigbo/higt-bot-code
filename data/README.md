# Datasets

All raw data is gitignored. Re-downloading is reproducible via `scripts/download_data.sh` (CTU-13, IoT-23) and the manual instructions below (MedBIoT).

## Layout

```
data/
  raw/                                 # original downloads — never modified
    CTU-13-Dataset/{1..13}/            # 13 scenarios; binetflow + pcap
    IoT-23 -> opt/.../IoTScenarios/    # symlink to the IoT-23 extraction
    opt/Malware-Project/BigDataset/    # actual IoT-23 path from the tar
      IoTScenarios/CTU-IoT-Malware-Capture-<N>-1/bro/conn.log.labeled
    medbiot/bulk/                      # MedBIoT bulk pcaps
  processed/                           # parsed flows (parquet) — Phase 2
  graphs/                              # PyG Data objects (.pt) — Phase 3
  inspection_logs/                     # Phase 1 inspection outputs (tracked in git)
```

**Note on IoT-23 path:** the tar packs with an `/opt/Malware-Project/BigDataset/IoTScenarios/` prefix. We symlink `data/raw/IoT-23` to that directory so paths in code stay readable. The symlink is created by `scripts/download_data.sh iot23`.

## CTU-13 — primary dataset (~1.9 GB)

- Source: <https://www.stratosphereips.org/datasets-ctu13>
- License: Creative Commons CC-BY
- Captured: 2011, CTU University, Czech Republic
- 13 scenarios, each one malware capture mixed with normal + background traffic
- **P2P-relevant scenarios: 9, 10, 11** (the only ones used for the headline P2P claim)

### File formats per scenario

- `*.pcap` — botnet traffic only (full mixed pcap is not released for privacy)
- `*.binetflow` — bidirectional NetFlow CSV with labels (this is what we parse)
- `*.biargus` — same flows in argus binary format (skip, use the CSV)
- `*.weblog`, `*.dns`, `*.duio` — auxiliary, not used

### Binetflow columns

`StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label`

### Label vocabulary

CTU-13 labels embed traffic type in the `Label` column as `flow=<class>-<scenario>-<detail>`. Three classes matter:

- `Background-...` → **background** (unlabeled, may secretly contain bot flows; exclude from training/eval).
- `From-Botnet-V<N>-...` and `To-Botnet-V<N>-...` → **bot**.
- `Normal-V<N>-...` → **benign**.

Anything not matching these three is also background.

### Quirks

- **Timezone:** timestamps are local Czech time, not UTC. Normalize to UTC during parsing.
- **Background = unlabeled, not benign.** Treat as a separate class; do not silently relabel as benign.
- **Single-packet flows** have undefined std/mean. Filter for stats but keep in counts.

## IoT-23 — real IoT botnet, attack execution phase (~8.7 GB lighter, ~20 GB full)

- Source: <https://www.stratosphereips.org/datasets-iot23>
- We use the **lighter** version (Zeek `conn.log.labeled` only, no pcaps).
- 23 scenarios, named `CTU-IoT-Malware-Capture-<N>-1`. Malware families: Mirai, Okiru, Hide and Seek, Hajime, Linux.Mirai, Trojan, Muhstik, etc.

### File format

`conn.log.labeled` is Zeek TSV with a trailing `tunnel_parents,label,detailed-label` triple. Standard Zeek columns:

`ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p, proto, service, duration, orig_bytes, resp_bytes, conn_state, ...`

### Label vocabulary

The `label` field is `Malicious` or `Benign`; the `detailed-label` adds family info like `PartOfAHorizontalPortScan`, `C&C`, `DDoS`, `Okiru-Attack`. Different scenarios use slightly different vocabularies — build a normalization map in `src/data/labels.py` when we get to Phase 2.

### Quirks

- **Timestamps in UTC** (Zeek convention) — no conversion needed. Joins with CTU-13 (Czech local) require care.
- **`conn.log.labeled` is in `<scenario>/bro/`** — not the scenario root.
- **Lighter version contains no pcaps.** Cannot re-parse with NFStream. Trust Zeek's output.

## MedBIoT — real malware, propagation + C&C phase

- Source: <https://cs.taltech.ee/research/data/medbiot/>
- 83 real + emulated IoT devices; malware: Mirai, BashLite, Torii
- Captures botnet **infection, propagation, and C&C communication** — the lifecycle window CTU-13 and IoT-23 mostly miss

### Manual download

The TalTech site does not provide a single tar. Use a browser to grab the **bulk pcaps** from:

<https://cs.taltech.ee/research/data/medbiot/bulk/>

Save them under `data/raw/medbiot/bulk/`. The file naming convention is:

- `<malware>_<traffic-type>_<device>.pcap`
- e.g., `mirai_mal_lock.pcap` = Mirai malicious traffic on lock devices
- e.g., `mirai_leg_lock.pcap` = legitimate traffic on lock devices *during* Mirai deployment (still benign — see the labelling note on the project page)

If you need fine-grained per-phase pcaps (C&C vs spreading), use:

<https://cs.taltech.ee/research/data/medbiot/fine-grained/>

We only need bulk for graph construction.

### Quirks

- **Phase 1 inspector uses `dpkt`, not NFStream.** NFStream's bundled nDPI binding fails on Apple Silicon macOS (flat-namespace symbol `_ndpi_category_get_name` not found, even after `brew install ndpi`). dpkt is pure-Python at the inner loop, fast enough (225 MB pcap in 6.5 s), and adequate for Phase 1 because labels come from filenames — no L7 classification needed. Phase 2 can reconsider on the Linux trainers if richer per-flow statistics are wanted.
- **`leg` traffic captured *during* malware deployment is still benign.** The infection is on a separate set of hosts. Do not relabel.
- **Device-type emulation** means many "devices" share characteristics. Treat each `(pcap_file, src_ip)` as a unique node identity, not just `src_ip`.

## Split documentation (preview, locked in Phase 3)

```
train  CTU-13 scenarios {3, 4, 5, 9, 10}    + IoT-23 {3-1, 7-1, 8-1, 17-1, 33-1}
val    CTU-13 scenarios {7, 11}             + IoT-23 {1-1, 9-1}
test   CTU-13 scenarios {2, 6, 8, 12}       + IoT-23 {20-1, 21-1, 35-1, 48-1}
       MedBIoT held out entirely as cross-domain test
```

Per-scenario chronological — never random across windows. See Phase 3 of `../../HiGT-Bot_Implementation_Plan.md`.
