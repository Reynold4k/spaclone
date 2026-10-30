from itertools import product

import numpy as np
import pandas as pd
import pytest

from spaclone import bcr

_SENSE = ["".join(c) for c in product("ACGT", repeat=3) if bcr.CODON_TABLE["".join(c)] != "*"]


def test_codon_table_and_classification():
    assert len(bcr.CODON_TABLE) == 64
    assert sorted(k for k, v in bcr.CODON_TABLE.items() if v == "*") == ["TAA", "TAG", "TGA"]
    assert bcr.CODON_TABLE["ATG"] == "M" and bcr.CODON_TABLE["TGG"] == "W"
    assert bcr.classify_mutation("TTA", 3, "G") == "S"  # Leu -> Leu
    assert bcr.classify_mutation("GCT", 1, "A") == "R"  # Ala -> Thr
    assert bcr.classify_mutation("TGG", 3, "A") == "stop"
    with pytest.raises(ValueError):
        bcr.classify_mutation("TT", 1, "A")


def test_selection_known_counts():
    rng = np.random.default_rng(0)
    germ = list("".join(rng.choice(_SENSE, 104)))
    germ[0:3] = list("TTA")  # FWR1 codon 1
    germ[78:81] = list("GCT")  # CDR1 codon 27 (nt 79-81)
    seq = list(germ)
    seq[2] = "G"  # TTA -> TTG: silent, FWR
    seq[78] = "A"  # GCT -> ACT: replacement, CDR
    air = pd.DataFrame({"sequence_alignment": ["".join(seq)], "germline_alignment": ["".join(germ)]})
    r = bcr.selection(air, n_perm=99, rng=0).table.set_index("region")
    assert (r.loc["FWR", "R"], r.loc["FWR", "S"]) == (0, 1)
    assert (r.loc["CDR", "R"], r.loc["CDR", "S"]) == (1, 0)
    assert 0.6 < r.loc["CDR", "expected_share_R"] < 0.85  # about 3/4 of random substitutions change the amino acid
    gapped = air.assign(sequence_alignment=lambda d: "..." + d.sequence_alignment.str[3:])
    r2 = bcr.selection(gapped, n_perm=19, rng=0).table.set_index("region")
    assert r2.loc["FWR", "mutations"] == 0  # the codon with the silent change is not sequenced


def _neutral(rng, n=80, k=8, cdr_r_bias=0.0):
    cdr = np.zeros(312, dtype=bool)
    cdr[78:114] = True
    cdr[165:195] = True
    rows = []
    for _ in range(n):
        germ = "".join(rng.choice(_SENSE, 104))
        s = list(germ)
        done = 0
        used = set()
        while done < k:
            p = int(rng.integers(0, 312))
            if p in used:
                continue
            alt = rng.choice([b for b in "ACGT" if b != germ[p]])
            c0 = p - p % 3
            cls = bcr.classify_mutation(germ[c0 : c0 + 3], p % 3 + 1, alt)
            if cls == "stop":
                continue
            if cdr[p] and cdr_r_bias and cls == "S" and rng.random() < cdr_r_bias:
                continue  # positive selection in CDRs: silent changes are lost
            s[p] = alt
            used.add(p)
            done += 1
        rows.append(("".join(s), germ))
    return pd.DataFrame(rows, columns=["sequence_alignment", "germline_alignment"])


def test_selection_calibration_under_neutral_mutation():
    rng = np.random.default_rng(1)
    reps = 100
    hits = 0
    for _ in range(reps):
        r = bcr.selection(_neutral(rng), n_perm=99, rng=rng).table.set_index("region")
        hits += r.loc["CDR", "p_value"] <= 0.05
    assert hits / reps <= 0.05 + 3.1 * np.sqrt(0.05 * 0.95 / reps)


def test_selection_power_determinism_and_groups():
    rng = np.random.default_rng(2)
    d = _neutral(rng, n=150, cdr_r_bias=0.9)
    r = bcr.selection(d, n_perm=499, rng=3, alternative="greater").table.set_index("region")
    assert r.loc["CDR", "sigma"] > 0.3 and r.loc["CDR", "p_value"] <= 0.01
    assert abs(r.loc["FWR", "sigma"]) < 0.3
    assert r.loc["CDR", "expected"] == pytest.approx(r.loc["CDR", "null_mean"], rel=0.05)
    a = bcr.selection(d, n_perm=49, rng=4, keep_null=True)
    b = bcr.selection(d, n_perm=49, rng=4, keep_null=True)
    np.testing.assert_array_equal(a.null, b.null)
    g = bcr.selection(d.assign(grp=np.arange(len(d)) % 3), by="grp", n_perm=19, rng=0)
    assert len(g.table) == 6 and set(g.table.region) == {"CDR", "FWR"}
    k2 = bcr.selection(d, kappa=2.0, n_perm=19, rng=0).table.set_index("region")
    assert k2.loc["CDR", "expected_share_R"] != pytest.approx(r.loc["CDR", "expected_share_R"])
    with pytest.raises(ValueError):
        bcr.selection(d, kappa=0)
