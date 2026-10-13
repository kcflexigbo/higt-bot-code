# P2P botnet data: inventory (2026-10-13)

## In-house data

Malware families per IoT-23 capture are from the official dataset page
(https://www.stratosphereips.org/datasets-iot23). CTU-13 families are from each scenario's README.

| Scenario | Family | P2P? | In corrected split | Windows | Bot host-windows | Notes |
|---|---|---|---|---|---|---|
| iot23-1-1 | Hide and Seek | yes (custom P2P) | train scenario (test part used) | 1,343 | 1,343 | one infected device; labelled bot traffic is 539,465 horizontal-scan flows + 8 C&C flows |
| iot23-9-1 | Linux.Hajime | yes (DHT/uTP) | train scenario (test part used) | 241 | 234 | one infected device; all 6.36 M labelled bot flows are TCP scans (ports 23/81); no P2P overlay traffic is labelled |
| ctu13-12 | NSIS.ay | yes (UDP P2P; ports 7600, 16001, 6881) | excluded (16 windows) | 16 | 28 | three infected hosts (147.32.84.165/.191/.192) + 6 external peers that originate labelled flows; 2,168 bot flows to 1,462 destinations |

Correction found on the way: `configs/split.yaml` calls held-out scenario iot23-8-1 "Hide-and-Seek"; the
official IoT-23 table lists 8-1 as **Hakai**. Hide and Seek is 1-1.

### Bot-to-bot structure (data/inspection_logs/p2p/graph_stats.json, raw_flow_check.txt)

Under the corrected labels (a host is a bot only if it originates bot flows) **no window graph in any
dataset, P2P captures included, contains a bot-bot edge**, apart from MedBIoT mirai spread (1,272
edges, the propagation phase between testbed devices) and two IoT-23 captures with 2 edges each.
In the raw flows of ctu13-12, 25 of 2,168 bot flows go between two bot originators; in iot23-1-1 and
iot23-9-1 there is a single infected device, so bot-bot edges cannot occur. The P2P peers of these
bots are outside the monitored network and are not labelled as bots unless they originate
labelled flows.

### Results on the in-house P2P families (main split, saved v3 test predictions; family_eval.json)

Every model (HiGT-Bot v2, XGB[x], RF, directed GINE, directed T-GINE, directed HiGT-Bot) reaches F1
1.000 / PR-AUC 1.000 with 0 FN and 0 FP on the test windows of iot23-1-1 (270 bot host-windows) and
iot23-9-1 (49). These captures are trivially separable by their scanning volume; they do not test
P2P detection.

## External data

| Dataset | Source | File | Size | sha256 | Content |
|---|---|---|---|---|---|
| ISOT Botnet 2010 | University of Victoria ISOT lab, https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/botnet-and-ransomware-detection-datasets/ (Google Drive link on that page) | ISOT_Botnet_DataSet_2010.tar.gz → ISOT_Botnet_DataSet_2010.pcap (11,394,249,522 bytes) | 2.8 GB compressed | see data/raw/p2p_isot/sha256.txt | Storm (Overnet UDP P2P) and Waledac (HTTP/fast-flux P2P) traces from the French Honeynet chapter, plus Zeus, replayed onto hosts of an Ericsson Lab background trace and merged with LBNL enterprise traces; malicious packets marked by synthetic MAC addresses (overview PDF, Table 2) |

Labelling rule for ISOT (src/data/parse_isot.py): a flow is bot if any packet carries one of the
malicious MACs; then the corrected host rule applies (bot = host that originates a bot flow in the
window, scripts/relabel_infected_src.py --any-dst). Bot hosts per the overview: 172.16.2.11 (Storm
UDP), 172.16.0.2 and 172.16.0.11 (Waledac spam), 172.16.0.12 (Storm spam), 172.16.2.12 (Zeus).
External Overnet/Waledac peers are labelled bot only if they originate marked flows.

Not used: CIC/ISCX Botnet 2014 (registration form), ISOT ransomware (signed agreement), the
University of Georgia Storm/Waledac/Kelihos/ZeroAccess traces (no public download found).
