"""Unit tests for Phase 2 parsers and the canonical schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.parse_ctu13 import parse_ctu13_scenario
from src.data.parse_iot23 import parse_iot23_scenario
from src.data.parse_medbiot import parse_medbiot_pcap
from src.data.schema import (
    FLOW_COLUMNS,
    SchemaError,
    coerce_to_schema,
    empty_flow_df,
    validate_flow_df,
)


# --------------------------------------------------------------------------- #
# Schema validator                                                            #
# --------------------------------------------------------------------------- #


def test_empty_flow_df_validates() -> None:
    df = empty_flow_df()
    assert list(df.columns) == FLOW_COLUMNS
    validate_flow_df(df)


def test_validate_rejects_missing_columns() -> None:
    df = empty_flow_df().drop(columns=["scenario"])
    with pytest.raises(SchemaError, match="Column mismatch"):
        validate_flow_df(df)


def test_validate_rejects_bad_label() -> None:
    df = empty_flow_df()
    df.loc[0] = [
        0, "test", "1.1.1.1", "2.2.2.2", 80, 80, "tcp",
        pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC"),
        1.0, 100, 0, 1, 0, np.nan, np.nan, np.nan, np.nan, "wrong", "x",
    ]
    df = coerce_to_schema(df)
    with pytest.raises(SchemaError, match="Invalid labels"):
        validate_flow_df(df)


def test_validate_rejects_duplicate_flow_id() -> None:
    df = empty_flow_df()
    base = [
        0, "test", "1.1.1.1", "2.2.2.2", 80, 80, "tcp",
        pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC"),
        1.0, 100, 0, 1, 0, np.nan, np.nan, np.nan, np.nan, "benign", "x",
    ]
    df.loc[0] = base
    df.loc[1] = base  # duplicate flow_id
    df = coerce_to_schema(df)
    with pytest.raises(SchemaError, match="not unique"):
        validate_flow_df(df)


# --------------------------------------------------------------------------- #
# CTU-13 parser                                                               #
# --------------------------------------------------------------------------- #


CTU13_HEADER = (
    "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,"
    "TotPkts,TotBytes,SrcBytes,Label"
)


@pytest.fixture
def tiny_ctu13(tmp_path: Path) -> Path:
    d = tmp_path / "10"
    d.mkdir()
    p = d / "capture.binetflow"
    rows = [
        # bot
        "2011-08-17 10:00:00.000000,0.5,tcp,10.0.0.5,1234,->,8.8.8.8,80,S_RA,0,0,5,500,300,flow=From-Botnet-V52-TCP",
        # benign with hex port
        "2011-08-17 10:00:02.000000,1.0,udp,10.0.0.6,0x0035,<->,1.1.1.1,53,CON,0,0,2,100,60,flow=Normal-V52-DNS",
        # background, zero-pkt — should be dropped
        "2011-08-17 10:00:04.000000,0.1,tcp,10.0.0.8,9000,->,8.8.8.8,9001,S_,0,0,0,0,0,flow=Background-Established",
    ]
    p.write_text(CTU13_HEADER + "\n" + "\n".join(rows) + "\n")
    return d


def test_parse_ctu13_basic(tiny_ctu13: Path) -> None:
    df = parse_ctu13_scenario(tiny_ctu13, scenario_id="ctu13-10")

    # Zero-pkt row dropped → 2 rows remain.
    assert len(df) == 2
    assert df["scenario"].unique().tolist() == ["ctu13-10"]
    assert df["flow_id"].is_unique
    assert df["start_time"].dt.tz is not None
    assert str(df["start_time"].dt.tz) == "UTC"

    # Labels include one bot, one benign.
    assert set(df["label"]) == {"bot", "benign"}

    # Hex port parsed correctly: 0x0035 → 53.
    benign = df[df["label"] == "benign"].iloc[0]
    assert int(benign["src_port"]) == 53

    # Bytes split: bot row had SrcBytes=300, TotBytes=500.
    bot = df[df["label"] == "bot"].iloc[0]
    assert int(bot["bytes_fwd"]) == 300
    assert int(bot["bytes_bwd"]) == 200

    # mean_iat / pkt_size NaN for CTU-13 (no per-packet info).
    assert df["mean_iat_ms"].isna().all()
    assert df["min_pkt_size"].isna().all()


def test_parse_ctu13_validates(tiny_ctu13: Path) -> None:
    df = parse_ctu13_scenario(tiny_ctu13)
    validate_flow_df(df)  # raises on schema violation


# --------------------------------------------------------------------------- #
# IoT-23 parser                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_iot23(tmp_path: Path) -> Path:
    scenario = tmp_path / "CTU-IoT-Malware-Capture-48-1" / "bro"
    scenario.mkdir(parents=True)
    p = scenario / "conn.log.labeled"
    header = (
        "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\t"
        "proto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\t"
        "local_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts\t"
        "orig_ip_bytes\tresp_pkts\tresp_ip_bytes\ttunnel_parents   label   detailed-label\n"
    )
    lines = [
        # malicious http
        "1551377734.188184\tC1\t192.168.1.200\t52724\t167.99.182.238\t80\t"
        "tcp\thttp\t1.98\t149\t119442\tSF\t-\t-\t0\tShAD\t174\t11698\t172\t247844\t-   Malicious   C&C-HeartBeat",
        # benign dns
        "1551377735.000000\tC2\t192.168.1.5\t52726\t8.8.8.8\t53\t"
        "udp\tdns\t0.05\t40\t100\tSF\t-\t-\t0\tDd\t1\t60\t1\t120\t-   Benign   -",
        # zero pkts — dropped
        "1551377736.000000\tC3\t192.168.1.6\t52727\t8.8.4.4\t443\t"
        "tcp\tssl\t10.0\t0\t0\tS0\t-\t-\t0\tS\t0\t0\t0\t0\t-   Benign   -",
    ]
    p.write_text(header + "\n".join(lines) + "\n")
    return tmp_path / "CTU-IoT-Malware-Capture-48-1"


def test_parse_iot23_basic(tiny_iot23: Path) -> None:
    df = parse_iot23_scenario(tiny_iot23)

    # Zero-pkt row dropped → 2 rows.
    assert len(df) == 2
    assert df["scenario"].iloc[0] == "iot23-48-1"
    assert set(df["label"]) == {"bot", "benign"}

    bot = df[df["label"] == "bot"].iloc[0]
    assert int(bot["bytes_fwd"]) == 149
    assert int(bot["bytes_bwd"]) == 119442
    assert int(bot["pkts_fwd"]) == 174
    assert int(bot["pkts_bwd"]) == 172
    assert bot["detailed_label"] == "C&C-HeartBeat"

    assert df["mean_iat_ms"].isna().all()


# --------------------------------------------------------------------------- #
# MedBIoT parser — run on a real small pcap (torii_mal_all, 24 MB)            #
# Skipped if the file isn't downloaded yet.                                   #
# --------------------------------------------------------------------------- #

TORII_PCAP = Path("data/raw/medbiot/bulk/raw_dataset/malware/torii_mal_all.pcap")


@pytest.mark.skipif(not TORII_PCAP.exists(), reason="torii_mal_all.pcap not downloaded")
def test_parse_medbiot_torii_real() -> None:
    df = parse_medbiot_pcap(TORII_PCAP)
    # From Phase 1 inspection: 102 bidirectional flows, all bot, family torii.
    assert len(df) == 102
    assert (df["label"] == "bot").all()
    assert df["detailed_label"].str.startswith("torii").all()
    assert df["scenario"].iloc[0] == "medbiot-torii_mal_all"

    # Per-packet stats SHOULD be populated for pcap-derived flows.
    has_iat = df["mean_iat_ms"].notna().sum()
    assert has_iat > 0  # at least some multi-packet flows
    assert df["min_pkt_size"].notna().all()
    assert df["max_pkt_size"].notna().all()
    # Reasonable sanity: min ≤ max within each flow.
    assert (df["min_pkt_size"] <= df["max_pkt_size"]).all()
