"""Tests of spaclone.bcr.lineage_workflow on synthetic TRUST4-like AIRR records with known lineages."""

import types

import numpy as np
import pandas as pd
import pytest

from spaclone import bcr, tl
from spaclone.bcr.lineage_workflow import LineageWorkflowWarning

# A small heavy-chain "germline": V (60 nt, ends with the conserved cysteine codon TGT at 57-59 -> the junction starts
# at V germline position 57), junction (Cys ... Trp), FWR4 (J germline after the junction).
V_GERM = "GAGGTCCAGCTGCAGCAGTCTGGACCTGAGCTGGTGAAGCCTGGGGCTTCAGTGAAGTGT"
V_GERM2 = V_GERM[:20] + "C" + V_GERM[21:30] + "A" + V_GERM[31:40] + "G" + V_GERM[41:]  # another allele, 3 sites
FWR4 = "GGCCAAGGCACCACTCTCACAGTCTCCTCAG"
LEADER = "ACGTTTGACC"  # 5' of the V gene in the contig
CONST = "GCCAAAACGACACCCCCATCTGTC"  # 3' of the J gene


def _junction(cdr3_nt: str) -> str:
    return "TGT" + cdr3_nt + "TGG"


def _aa(nt: str) -> str:
    table = {
        "TGT": "C",
        "TGG": "W",
        "GCA": "A",
        "AGA": "R",
        "GAC": "D",
        "TAC": "Y",
        "TTC": "F",
        "GGC": "G",
        "AAC": "N",
        "TCC": "S",
        "CAA": "Q",
    }
    return "".join(table.get(nt[i : i + 3], "X") for i in range(0, len(nt) - 2, 3))


def _record(sid, x, y, v_seq, junction, *, reads=50, v_call="IGHV1-1*01", c_call="IGHM*01", lead=0, j_call="IGHJ1*01",
            germ=V_GERM):  # fmt: skip
    """One TRUST4-like AIRR record: contig = leader + V (from germline position ``lead``) + junction + FWR4 + C."""
    v_part = v_seq[lead:57]
    contig = LEADER + v_part + junction + FWR4 + CONST
    v_start = len(LEADER)
    v_al = v_seq[lead:60]  # V alignment runs into the cysteine codon
    g_al = germ[lead:60]
    j_start = v_start + len(v_part) + len(junction) - 6  # J covers the last 6 junction nt
    j_len = 6 + len(FWR4)
    rest = len(contig) - j_start - j_len
    return {
        "sequence_id": sid,
        "cell_id": f"{int(x * 2)}-{int(y * 2)}",
        "spot_id": f"{int(x * 2)}-{int(y * 2)}",
        "x_um": float(x),
        "y_um": float(y),
        "locus": "IGH",
        "v_call": v_call,
        "j_call": j_call,
        "c_call": c_call,
        "junction": junction,
        "junction_aa": _aa(junction),
        "productive": True,
        "consensus_count": float(reads),
        "sequence": contig,
        "v_sequence_alignment": v_al,
        "v_germline_alignment": g_al,
        "v_germline_start": float(lead),
        "v_cigar": f"{v_start}S{lead}N{len(v_al)}M{len(contig) - v_start - len(v_al)}S" if lead else
        f"{v_start}S{len(v_al)}M{len(contig) - v_start - len(v_al)}S",
        "j_cigar": f"{j_start}S{j_len}M{rest}S",
        "sequence_alignment": v_part + junction + FWR4,
        "germline_alignment": germ[lead:57] + junction + FWR4,
    }  # fmt: skip


def _mutate(seq: str, sites: dict[int, str]) -> str:
    s = list(seq)
    for p, b in sites.items():
        s[p] = b
    return "".join(s)


def _lineage_table(seed: int = 0) -> pd.DataFrame:
    """Clone A: 8 tips of a planted lineage (shared + private mutations, each in two spots with >= 30 reads), plus a
    noise variant seen once; clone B: one junction nucleotide away from A (same V, J, length: related, not exact);
    clone C: 3 distinct sequences only (no tree)."""
    rng = np.random.default_rng(seed)
    jA = _junction("GCAAGAGACTACTTCGGCAAC")
    jB = _junction("GCAAGAGACTACTTCGGCGAC")  # one nt from jA (CARDYFGDW: same length, 96% identity)
    jC = _junction("GCAAGAGGCTCCAACTAC")
    rows = []
    lineage = {
        "a1": {10: "A"},
        "a2": {10: "A", 20: "G"},
        "a3": {10: "A", 20: "G", 33: "A"},
        "a4": {10: "A", 40: "G"},
        "a5": {5: "G"},
        "a6": {5: "G", 25: "T"},
        "a7": {5: "G", 25: "T", 50: "G"},
        "a8": {},
    }
    k = 0
    for name, sites in lineage.items():
        seq = _mutate(V_GERM, sites)
        for rep in range(2):  # every sequence in two spots far apart (two independent assemblies)
            x, y = 100 + 37 * k + 400 * rep, 200 + 11 * k
            c = "IGHG1*01" if 25 in sites else "IGHM*01"
            rows.append(_record(f"A_{name}_{rep}", x, y, seq, jA, reads=40 + rep, c_call=c))
            k += 1
    noisy = _mutate(V_GERM, {10: "A", 44: "A"})  # 44:A seen in one spot only -> reverted
    rows.append(_record("A_noise", 900, 900, noisy, jA, reads=200))
    for i in range(3):
        rows.append(_record(f"B_{i}", 1300 + 40 * i, 300, _mutate(V_GERM, {10: "A", 52: "G"}), jB, reads=35))
    for i, sites in enumerate([{}, {7: "T"}, {7: "T", 8: "C"}]):
        for rep in range(2):
            rows.append(_record(f"C_{i}_{rep}", 2000 + 30 * i + 300 * rep, 800, _mutate(V_GERM, sites), jC,
                                reads=60))  # fmt: skip
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=int(rng.integers(1000))).reset_index(drop=True)


@pytest.fixture(scope="module")
def table():
    return _lineage_table()


@pytest.fixture(scope="module")
def result(table):
    return bcr.lineage_workflow(table, compartments=False, min_tips=5, rng=1)


def test_exact_and_related_groupings(result, table):
    rec = result.records.set_index("sequence_id")
    assert rec.loc["A_a1_0", "clone_exact"] != rec.loc["B_0", "clone_exact"]
    assert rec.loc["A_a1_0", "clone_related"] == rec.loc["B_0", "clone_related"]
    assert rec.loc["A_a1_0", "clone_related"] != rec.loc["C_0_0", "clone_related"]
    assert len(rec) == len(table)


def test_planted_mutations_and_noise(result):
    tips = result.tips["exact"]
    a = tips[tips.tree_id.str.contains("CARDYFGNW")]
    assert len(a) > 0
    rec = result.records.set_index("sequence_id")
    # planted mutations counted exactly (every planted variant is in two spots with >= 30 reads)
    expect = {"A_a1_0": 1, "A_a2_0": 2, "A_a3_0": 3, "A_a4_1": 2, "A_a5_0": 1, "A_a6_1": 2, "A_a7_0": 3, "A_a8_0": 0}
    for sid, n in expect.items():
        assert rec.loc[sid, "mutations_exact"] == n, sid
    # the variant seen in one spot is an assembly error: reverted, so the noisy record equals a1
    assert rec.loc["A_noise", "mutations_exact"] == 1
    assert rec.loc["A_noise", "sequence_exact"] == rec.loc["A_a1_0", "sequence_exact"]


def test_selection_rules(result):
    cl = result.clones["exact"].set_index("clone_id")
    a = [c for c in cl.index if c.endswith("CARDYFGNW")][0]
    c = [c for c in cl.index if c.endswith("CARGSNYW")][0]
    assert cl.loc[a, "selected"]
    assert cl.loc[a, "distinct_seqs"] == 8
    assert cl.loc[a, "mutation_sites"] == 7
    assert not cl.loc[c, "selected"]  # 3 distinct sequences < 5
    assert a in result.trees["exact"] and c not in result.trees["exact"]


def test_tips_one_per_locus_and_sequence(result):
    t = result.tips["exact"]
    assert not t.duplicated(["tree_id", "locus_id", "sequence"]).any()
    a = t[t.tree_id.str.endswith("CARDYFGNW")]
    assert len(a) == 17  # 8 sequences x 2 spots + the cleaned noisy record in its own spot
    assert set(a.isotype) <= {"IgM", "IgG"}
    assert (a.loc[a.sequence == a.sequence.iloc[0], "mutations"].nunique()) == 1


def test_trees_and_selection_table(result):
    sel = result.selection[result.selection.grouping == "exact"].set_index("tree_id")
    a = [c for c in sel.index if c.endswith("CARDYFGNW")][0]
    assert sel.loc[a, "eligible"] and sel.loc[a, "drawn"] and sel.loc[a, "rank"] == 1
    tree = result.trees["exact"][a]
    assert "Germline" in tree.tips()
    # tree depth of a tip equals its mutation count on a parsimony tree of these clean sequences
    tt = result.tree_tips(a, grouping="exact")
    assert np.allclose(tt.depth.to_numpy(), tt.mutations.to_numpy())
    assert result.newick("exact")[a].endswith(";")


def test_related_members(result):
    sel = result.selection
    rel = sel[(sel.grouping == "related") & sel.tree_id.str.contains("CARDYFGNW")]
    assert len(rel) == 1
    m = result.members[result.members.tree_id == rel.tree_id.iloc[0]]
    assert len(m) == 2
    assert m.n_tips.iloc[0] >= m.n_tips.iloc[1]


def test_frame_check_and_misfit(table):
    bad = table[table.sequence_id == "A_a2_0"].copy()
    bad["sequence_id"] = "A_stop"
    contig = bad.sequence.iloc[0]
    pos = len(LEADER) + 30  # an in-frame stop codon inside V
    bad["sequence"] = contig[:pos] + "TAA" + contig[pos + 3 :]
    other = table[table.sequence_id == "A_a8_0"].copy()
    other["sequence_id"] = "A_other_allele"
    seq = V_GERM2
    rec = _record("A_other_allele", 50, 50, seq, other.junction.iloc[0], v_call="IGHV1-1*02", germ=V_GERM2)
    df = pd.concat([table, bad, pd.DataFrame([rec])], ignore_index=True)
    res = bcr.lineage_workflow(df, compartments=False, min_tips=5, rng=1, groupings=("exact",))
    r = res.records.set_index("sequence_id")
    assert not r.loc["A_stop", "usable"]
    assert r.loc["A_other_allele", "usable"]
    assert r.loc["A_other_allele", "misfit_exact"]  # its own allele fits 3 positions better than the consensus


def test_truncated_records_support_but_are_no_tips(table):
    tr = table[table.sequence_id == "A_a3_0"].copy()
    rows = []
    for i in range(2):
        r = _record(f"A_trunc_{i}", 3000 + 50 * i, 50, _mutate(V_GERM, {10: "A", 20: "G", 33: "A", 48: "C"}),
                    tr.junction.iloc[0], lead=30, reads=90)  # fmt: skip
        rows.append(r)
    df = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    res = bcr.lineage_workflow(df, compartments=False, min_tips=5, rng=1, groupings=("exact",))
    r = res.records.set_index("sequence_id")
    assert r.loc["A_trunc_0", "usable"] and not r.loc["A_trunc_0", "full_length"]
    assert not res.tips["exact"].record_ids.str.contains("A_trunc").any()


def test_locations_from_hotspots_and_compartments(table):
    grid = tl.Grid((0.0, 0.0), 10.0, (400, 120))
    hot = np.zeros(grid.shape, dtype=np.int64)
    hot[:30, :] = 1  # x < 300 um: a CSR hotspot
    comp_codes = np.zeros(grid.shape, dtype=np.int16)  # T zone everywhere ...
    comp_codes[100:, :] = 2  # ... red pulp from x = 1000 um
    comp = tl.Raster(comp_codes, grid, categories=("T zone", "follicle", "red pulp"))
    res = bcr.lineage_workflow(table, hotspots=tl.Raster(hot, grid), compartments=comp, min_tips=5, rng=1)
    loci = res.loci
    x = loci.x_um.to_numpy()
    assert (loci.location[x < 300] == "CSR hotspot").all()
    assert (loci.location[(x >= 300) & (x < 1000)] == "T zone").all()
    assert (loci.location[(x >= 1000) & (x < 4000)] == "red pulp").all()
    t = res.tips["exact"]
    assert set(t.location) <= {"CSR hotspot", "T zone", "red pulp"}
    assert res.location_classes == ("CSR hotspot", "T zone", "red pulp", "follicle")
    assert res.params["compartments_from"] == "given"


def test_compartments_from_gex():
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(3)
    n = 4000
    xy = rng.uniform(0, 1000, size=(n, 2))
    lab = np.where(xy[:, 0] < 400, "T_cells", np.where(xy[:, 0] < 800, "B_cells", "Erythrocyte"))
    gex = ad.AnnData(obs=pd.DataFrame({"cell2loc_anno": lab}, index=[f"s{i}" for i in range(n)]),
                     obsm={"spatial": xy})  # fmt: skip
    df = _lineage_table()
    df["x_um"] = np.clip(df.x_um / 4.0, 0, 999)
    res = bcr.lineage_workflow(df, gex, min_tips=5, rng=1, groupings=("exact",))
    assert res.maps["compartments"] is not None and res.maps["tissue"] is not None
    loc = res.loci.set_index("locus_id").location
    x = res.loci.set_index("locus_id").x_um
    assert (loc[x < 350] == "T zone").all()
    assert (loc[(x > 450) & (x < 750)] == "follicle").all()
    assert (loc[x > 850] == "red pulp").all()
    assert res.maps["rotation"] is not None
    assert res.location_classes == ("T zone", "red pulp", "follicle") and res.params["compartments_from"] == "gex"
    # a hotspot workflow result keeps its transcriptome: lineage_workflow(result) needs no gex=
    wf = types.SimpleNamespace(records=df, gex=gex, csr=None, params={"record_level": "contig"})
    res2 = bcr.lineage_workflow(wf, min_tips=5, rng=1, groupings=("exact",))
    pd.testing.assert_frame_equal(res2.loci, res.loci)
    assert res2.location_classes == res.location_classes


def test_without_a_transcriptome_locations_are_unknown_and_it_says_so(table):
    with pytest.warns(LineageWorkflowWarning, match=r"no transcriptome.*gex=.*compartments=False"):
        res = bcr.lineage_workflow(table, min_tips=5, rng=1, groupings=("exact",))
    assert set(res.loci.location) == {"unknown"} and res.location_classes == ("unknown",)
    assert res.params["compartments_from"] is None and res.maps["compartments"] is None
    quiet = bcr.lineage_workflow(table, compartments=False, min_tips=5, rng=1, groupings=("exact",))
    pd.testing.assert_frame_equal(quiet.loci, res.loci)


def test_read_floor_auto_and_warning(table):
    assert bcr.lineage_workflow(table, compartments=False, min_tips=5, rng=1).params["min_reads"] == 2
    low = table.copy()
    n = int(0.6 * len(low))
    low.loc[low.index[:n], "consensus_count"] = 1.0
    with pytest.warns(LineageWorkflowWarning, match=rf"min_reads=2 removed {n} of {len(low)} records .*min_reads=1"):
        bcr.lineage_workflow(low, compartments=False, min_tips=5, rng=1, groupings=("exact",))
    low.attrs = {"spaclone": {"level": "molecule"}}  # one UMI molecule per record: no read floor
    res = bcr.lineage_workflow(low, compartments=False, min_tips=5, rng=1, groupings=("exact",))
    assert res.params["min_reads"] == 1 and res.params["counts"]["reads_ge_1"] == len(table)


def test_determinism_and_row_order(table):
    a = bcr.lineage_workflow(table, compartments=False, min_tips=5, rng=7)
    b = bcr.lineage_workflow(table.iloc[::-1].reset_index(drop=True), compartments=False, min_tips=5, rng=7)
    for g in ("exact", "related"):
        assert a.newick(g) == b.newick(g)
        cols = ["tree_id", "locus_id", "sequence", "mutations", "isotype", "tree_label"]
        ta = a.tips[g][cols].sort_values(["tree_id", "locus_id", "sequence"]).reset_index(drop=True)
        tb = b.tips[g][cols].sort_values(["tree_id", "locus_id", "sequence"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(ta, tb)
    pd.testing.assert_frame_equal(a.selection, b.selection)


def test_errors(table):
    with pytest.raises(ValueError, match=r"no v_sequence_alignment\. Lineage trees .*igblastn -outfmt 19"):
        bcr.lineage_workflow(table.drop(columns=["v_sequence_alignment"]), compartments=False)
    clone_table = table.drop(columns=[c for c in table.columns if "sequence" in c or "alignment" in c])
    with pytest.raises(ValueError, match=r"clone tables .* cannot give lineage trees"):  # e.g. AIR-SPACE clone tables
        bcr.lineage_workflow(clone_table, compartments=False)
    with pytest.raises(KeyError, match="lacks"):
        bcr.lineage_workflow(table.drop(columns=["junction_aa"]), compartments=False)
    with pytest.raises(ValueError, match="grouping"):
        bcr.lineage_workflow(table, compartments=False, groupings=("clonotype",))
    with pytest.raises(ValueError, match="order"):
        bcr.lineage_workflow(table, compartments=False, order="random")
    with pytest.raises(ValueError, match=r"no heavy-chain record.*IGH 0; .*min_reads=2, auto"):
        bcr.lineage_workflow(table.assign(locus="IGK"), compartments=False)
    with pytest.raises(ValueError, match=r"reads >= 1000 0; min_reads=1000, given"):
        bcr.lineage_workflow(table, compartments=False, min_reads=1000)
