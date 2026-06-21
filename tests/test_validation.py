import pandas as pd

from src.evaluation.validation import group_well_folds, known_zone_mask, target_zone_mask


def test_target_zone_mask_selects_missing_tvt_input_rows():
    df = pd.DataFrame({"TVT_input": [1.0, 2.0, None], "TVT": [1.0, 2.0, 3.0]})

    assert target_zone_mask(df).tolist() == [False, False, True]
    assert known_zone_mask(df).tolist() == [True, True, False]


def test_group_well_folds_partition_wells_without_overlap():
    folds = group_well_folds(["a", "b", "c", "d", "e"], n_splits=3)

    held_out = []
    for train_wells, valid_wells in folds:
        assert set(train_wells).isdisjoint(valid_wells)
        held_out.extend(valid_wells)

    assert sorted(held_out) == ["a", "b", "c", "d", "e"]
