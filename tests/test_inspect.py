"""Unit tests for src/data/inspect.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.inspect import (
    CTU13_BINETFLOW_COLUMNS,
    load_ctu13_binetflow,
    load_iot23_conn,
    normalize_ctu13_label,
    normalize_iot23_label,
)


@pytest.mark.parametrize("raw,expected", [
    ("flow=Background-Established", "background"),
    ("flow=Background-attempt-cmpgw-CVUT", "background"),
    ("flow=From-Botnet-V52-1-TCP-CC42-Custom-Encryption", "bot"),
    ("flow=To-Botnet-V52-UDP-Attempt", "bot"),
    ("flow=Normal-V52-Stribika-Web", "benign"),
    ("flow=Normal-V42-Grill", "benign"),
    ("", "background"),
    (None, "background"),
])
def test_normalize_label(raw: str, expected: str) -> None:
    assert normalize_ctu13_label(raw) == expected


@pytest.fixture
def tiny_binetflow(tmp_path: Path) -> Path:
    """A 6-row binetflow file: 2 bot, 2 benign, 2 background."""
    p = tmp_path / "tiny.binetflow"
    rows = [
        # bot 1
        ("2011-08-17 10:00:00.000000", "0.5", "tcp", "10.0.0.5", "1234",
         "->", "8.8.8.8", "80", "S_RA", "0", "0", "5", "500", "300",
         "flow=From-Botnet-V52-TCP"),
        # bot 2
        ("2011-08-17 10:00:01.000000", "0.5", "tcp", "10.0.0.5", "1235",
         "->", "8.8.4.4", "80", "S_RA", "0", "0", "3", "200", "120",
         "flow=To-Botnet-V52-TCP"),
        # benign 1
        ("2011-08-17 10:00:02.000000", "1.0", "udp", "10.0.0.6", "53",
         "<->", "1.1.1.1", "53", "CON", "0", "0", "2", "100", "60",
         "flow=Normal-V52-DNS"),
        # benign 2
        ("2011-08-17 10:00:03.000000", "1.0", "tcp", "10.0.0.7", "443",
         "->", "1.1.1.1", "443", "SA_FA", "0", "0", "10", "1500", "800",
         "flow=Normal-V52-HTTPS"),
        # background 1
        ("2011-08-17 10:00:04.000000", "0.1", "tcp", "10.0.0.8", "9000",
         "->", "8.8.8.8", "9001", "S_", "0", "0", "1", "60", "60",
         "flow=Background-Established"),
        # background 2
        ("2011-08-17 10:00:05.000000", "0.1", "icmp", "10.0.0.9", "0",
         "  ?>", "8.8.8.8", "0", "URP", "0", "0", "1", "84", "84",
         "flow=Background-other"),
    ]
    df = pd.DataFrame(rows, columns=CTU13_BINETFLOW_COLUMNS)
    df.to_csv(p, index=False)
    return p


def test_load_ctu13_binetflow_counts(tiny_binetflow: Path) -> None:
    df = load_ctu13_binetflow(tiny_binetflow)
    assert len(df) == 6
    counts = df["class"].value_counts().to_dict()
    assert counts["bot"] == 2
    assert counts["benign"] == 2
    assert counts["background"] == 2


def test_load_ctu13_binetflow_dtypes(tiny_binetflow: Path) -> None:
    df = load_ctu13_binetflow(tiny_binetflow)
    assert df["StartTime"].dtype.kind == "M"  # datetime64
    assert df["SrcAddr"].dtype == object


def test_load_rejects_bad_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.binetflow"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="Unexpected columns"):
        load_ctu13_binetflow(bad)


# --------------------------------------------------------------------------- #
# IoT-23 Zeek conn.log.labeled                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw,expected", [
    ("Malicious", "bot"),
    ("Benign", "benign"),
    ("  Malicious  ", "bot"),  # leading/trailing whitespace
    ("malicious", "background"),  # case-sensitive — Stratosphere uses capitalized
    ("-", "background"),
    ("", "background"),
    (None, "background"),
])
def test_normalize_iot23_label(raw: str, expected: str) -> None:
    assert normalize_iot23_label(raw) == expected


@pytest.fixture
def tiny_iot23(tmp_path: Path) -> Path:
    """A 4-row IoT-23 conn.log.labeled file: 2 Malicious, 1 Benign, 1 unlabeled.

    Mirrors the real format quirk: Zeek's original 21 columns are tab-separated,
    but the appended `label` and `detailed-label` use 3-space separation.
    """
    p = tmp_path / "conn.log.labeled"
    header = (
        "#separator \\x09\n"
        "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\t"
        "proto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\t"
        "local_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts\t"
        "orig_ip_bytes\tresp_pkts\tresp_ip_bytes\ttunnel_parents   label   detailed-label\n"
    )
    # Each line: 21 tab-separated original columns, then 3-space-separated label triple.
    lines = [
        "1551377734.188184\tCabc1\t192.168.1.200\t52724\t167.99.182.238\t80\t"
        "tcp\thttp\t1.978591\t149\t119442\tSF\t-\t-\t0\tShADadttfF\t174\t11698\t"
        "172\t247844\t-   Malicious   C&C-HeartBeat",
        "1551377735.000000\tCabc2\t192.168.1.200\t52725\t167.99.182.238\t80\t"
        "tcp\thttp\t2.0\t150\t100000\tSF\t-\t-\t0\tShADadttfF\t100\t8000\t"
        "100\t100000\t-   Malicious   PartOfAHorizontalPortScan",
        "1551377736.000000\tCabc3\t192.168.1.5\t52726\t8.8.8.8\t53\t"
        "udp\tdns\t0.05\t40\t100\tSF\t-\t-\t0\tDd\t1\t60\t1\t120\t-   Benign   -",
        "1551377737.000000\tCabc4\t192.168.1.6\t52727\t8.8.4.4\t443\t"
        "tcp\tssl\t10.0\t1500\t8000\tSF\t-\t-\t0\tShADadFf\t30\t3000\t"
        "25\t10000\t-   -   -",
    ]
    p.write_text(header + "\n".join(lines) + "\n")
    return p


def test_load_iot23_conn_counts(tiny_iot23: Path) -> None:
    df = load_iot23_conn(tiny_iot23)
    assert len(df) == 4
    counts = df["class"].value_counts().to_dict()
    assert counts["bot"] == 2
    assert counts["benign"] == 1
    assert counts.get("background", 0) == 1


def test_load_iot23_conn_dtypes(tiny_iot23: Path) -> None:
    df = load_iot23_conn(tiny_iot23)
    assert df["ts"].dtype.kind == "M"
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["id.orig_h"].dtype == object


def test_load_iot23_conn_detailed_label(tiny_iot23: Path) -> None:
    df = load_iot23_conn(tiny_iot23)
    bot_detail = df.loc[df["class"] == "bot", "detailed-label"].tolist()
    assert "C&C-HeartBeat" in bot_detail
    assert "PartOfAHorizontalPortScan" in bot_detail
