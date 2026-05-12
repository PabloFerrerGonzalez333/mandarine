from mandarine.config import load_dataset_config
from mandarine.data.splits import create_split_directories


def test_split_is_deterministic() -> None:
    config = load_dataset_config("configs/modern/dataset/mandarins_cropped.yaml")
    split_a, rows_a = create_split_directories(config, 13)
    split_b, rows_b = create_split_directories(config, 13)

    assert split_a.split_root == split_b.split_root
    assert rows_a == rows_b


def test_train_never_contains_abnormal_images() -> None:
    config = load_dataset_config("configs/modern/dataset/mandarins_cropped.yaml")
    _, rows = create_split_directories(config, 42)
    train_rows = [row for row in rows if row["split"] == "train"]
    assert train_rows
    assert {row["label"] for row in train_rows} == {"good"}


def test_expected_counts_for_each_seed() -> None:
    config = load_dataset_config("configs/modern/dataset/mandarins_cropped.yaml")
    _, rows = create_split_directories(config, 23)
    assert len([row for row in rows if row["split"] == "train"]) == config.train_count
    assert len([row for row in rows if row["split"] == "val" and row["label"] == "good"]) == config.val_good_count
    assert len([row for row in rows if row["split"] == "val" and row["label"] == "bad"]) == config.val_bad_count
    assert len([row for row in rows if row["split"] == "test" and row["label"] == "good"]) == config.test_good_count
    assert len([row for row in rows if row["split"] == "test" and row["label"] == "bad"]) == config.test_bad_count
