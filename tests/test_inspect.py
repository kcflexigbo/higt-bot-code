"""Unit tests for src/data/inspect.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.inspect import (
    CTU13_BINETFLOW_COLUMNS,
    load_ctu13_binetflow,
    normalize_ctu13_label,
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
