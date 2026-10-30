import numpy as np
import pandas as pd
import pytest

from spaclone.pp import knee_point, qc_metrics, read_support_curve


def _toy():
    return pd.DataFrame(
        {
            "spot_id": ["a", "a", "b", "c", "c", "d", "e", "e"],
            "x_um": [0.0, 0, 10, 20, 20, 30, 40, 40],
            "y_um": [0.0] * 8,
            "locus": ["IGH", "IGK", "IGH", "TRA", "TRB", "IGH", "IGK", "IGK"],
            "clone_id": ["h1", "k1", "h1", "a1", "b1", "h2", "k1", "k2"],
            "junction_aa": ["CARDYW", "CQQYF", "CAR*YW", "CAVRDSNYQLIW", "CASSF", "CARDYW", "CQQYF", "CQQLF"],
            "productive": ["T", "T", "T", "F", "T", "TRUE", "1", "T"],
            "consensus_count": [5, 1, 9, 3, 2, np.nan, 2, 7],
            "in_tissue": [True, True, True, True, False, True, True, True],
        }
    )


def test_qc_metrics_group_table_known_values():
    t, sp = qc_metrics(_toy())
    t = t.set_index("locus")
    assert t.n_records.to_dict() == {"IGH": 3, "IGK": 3, "TRA": 1, "TRB": 1}
    assert t.loc["IGH", "n_spots"] == 3 and t.loc["IGK", "n_spots"] == 2
    assert t.loc["IGH", "n_clones"] == 2 and t.loc["IGK", "n_clones"] == 2
    assert t.loc["IGH", "reads_total"] == 14  # NaN reads ignored
    assert t.loc["IGK", "reads_q50"] == 2 and t.loc["IGK", "share_single_read"] == pytest.approx(1 / 3)
    assert t.loc["IGH", "valid_share"] == pytest.approx(2 / 3)
    assert t.loc["TRA", "productive_share"] == 0 and t.loc["IGK", "productive_share"] == 1
    assert t.locus_share.sum() == pytest.approx(1.0) and t.loc["IGH", "locus_share"] == pytest.approx(3 / 8)
    # IGK: spot a holds k1, spot e holds k1 and k2 -> clones per spot 1 and 2; k1 on 2 spots, k2 on 1
    assert t.loc["IGK", "clones_per_spot_mean"] == 1.5 and t.loc["IGK", "share_spots_one_clone"] == 0.5
    assert t.loc["IGK", "spots_per_clone_max"] == 2 and t.loc["IGK", "spots_per_clone_mean"] == 1.5
    assert t.loc["TRB", "off_tissue_share"] == 1 and t.loc["IGH", "off_tissue_share"] == 0
    assert (t.sink_spot_share == 0).all()


def test_qc_metrics_spot_table_and_sinks():
    air = _toy()
    sink = pd.DataFrame(
        {
            "spot_id": "z",
            "x_um": 99.0,
            "y_um": 0.0,
            "locus": "IGK",
            "clone_id": [f"s{i}" for i in range(20)],
            "junction_aa": "CQQYF",
            "productive": "T",
            "consensus_count": 1.0,
            "in_tissue": True,
        }
    )
    t, sp = qc_metrics(pd.concat([air, sink], ignore_index=True), by=())
    sp = sp.set_index("spot_id")
    assert sp.loc["z", "n_clones"] == 20 and bool(sp.loc["z", "is_sink"])
    assert not sp.drop(index="z").is_sink.any()
    assert sp.loc["a", "n_records"] == 2 and sp.loc["a", "n_loci"] == 2 and sp.loc["a", "n_IGH"] == 1
    assert sp.loc["b", "valid_share"] == 0 and sp.loc["e", "n_reads"] == 9
    assert t.sink_spot_share.iloc[0] == pytest.approx(1 / 6)
    assert t.sink_record_share.iloc[0] == pytest.approx(20 / 28)
    # threshold is a parameter
    t2, sp2 = qc_metrics(pd.concat([air, sink], ignore_index=True), by=(), sink_min_clones=21)
    assert not sp2.is_sink.any()


def test_qc_metrics_spots_are_library_specific_and_order_invariant():
    air = _toy()
    two = pd.concat([air.assign(library_id="L1"), air.assign(library_id="L2")], ignore_index=True)
    t, sp = qc_metrics(two)
    assert set(t.library_id) == {"L1", "L2"} and len(sp) == 10
    perm = two.sample(frac=1.0, random_state=3)
    t2, sp2 = qc_metrics(perm)
    pd.testing.assert_frame_equal(t, t2, check_exact=False)
    cols = ["library_id", "spot_id", "n_records", "n_clones", "n_reads"]
    pd.testing.assert_frame_equal(sp[cols], sp2[cols])


def test_qc_metrics_missing_columns_and_errors():
    air = _toy()[["spot_id", "locus"]]
    t, sp = qc_metrics(air)
    assert np.isnan(t.valid_share).all() and np.isnan(t.reads_total).all() and (t.n_clones == 0).all()
    with pytest.raises(KeyError):
        qc_metrics(air, by="library_id")
    with pytest.raises(KeyError):
        qc_metrics(_toy(), tissue_key="tissue")
    with pytest.raises(ValueError):
        qc_metrics(_toy(), quantiles=(1.5,))


def test_knee_point_finds_the_gap_between_the_low_read_cloud_and_deep_records():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        low = rng.geometric(0.75, 7000)  # 1-2 reads mostly: the low-read cloud
        high = np.round(rng.lognormal(np.log(40), 0.4, 3000))  # deep, well-supported records
        k = knee_point(np.r_[low, high])
        assert low.max() <= k.inflection <= high.min() + 1  # the steepest drop sits in the gap
        assert k.inflection <= k.knee <= np.median(high) * 1.25  # the knee is the conservative end
    k = knee_point(np.random.default_rng(1).geometric(0.5, 5000))  # no deep component: small values
    assert k.knee <= 6
    assert np.isnan(knee_point([1, 1, 2, 2]).knee)  # fewer than three distinct values
    assert np.isnan(knee_point([]).inflection)


def test_read_support_curve_and_its_knee():
    rng = np.random.default_rng(5)
    low = rng.geometric(0.75, 7000)
    high = np.round(rng.lognormal(np.log(40), 0.4, 3000))
    air = pd.DataFrame({"spot_id": np.arange(10000).astype(str), "consensus_count": np.r_[low, high].astype(float)})
    curve, knee = read_support_curve(air, by=())
    assert curve.records.iloc[0] == 10000 and (np.diff(curve.records) <= 0).all()
    assert curve.share_spots.iloc[0] == 1.0
    k = knee_point(air.consensus_count)
    assert knee.knee_reads.iloc[0] == k.knee and knee.inflection_reads.iloc[0] == k.inflection
    assert knee.share_records.iloc[0] == pytest.approx((air.consensus_count >= k.knee).mean())


def test_read_support_curve_groups_and_floors():
    air = _toy()
    curve, knee = read_support_curve(air, by="locus", floors=[1, 2, 5])
    igk = curve[curve.locus == "IGK"].set_index("min_reads")
    assert igk.records.to_dict() == {1.0: 3, 2.0: 2, 5.0: 1}
    assert igk.spots.to_dict() == {1.0: 2, 2.0: 1, 5.0: 1}
    assert set(knee.locus) == {"IGH", "IGK", "TRA", "TRB"}
    with pytest.raises(KeyError):
        read_support_curve(air.drop(columns="consensus_count"))
