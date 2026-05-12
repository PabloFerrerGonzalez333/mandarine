from mandarine.config import (
    load_benchmark_config,
    load_dataset_config,
    load_final_train_config,
    load_model_config,
)


def test_dataset_config_resolves_real_paths() -> None:
    config = load_dataset_config("configs/modern/dataset/mandarins_cropped.yaml")
    assert config.source_root.exists()
    assert (config.source_root / config.normal_dir).exists()
    assert (config.source_root / config.abnormal_dir).exists()


def test_model_configs_are_loadable() -> None:
    benchmark = load_benchmark_config("configs/modern/benchmark_cpu.yaml")
    model_names = [load_model_config(path).name for path in benchmark.model_configs]
    assert model_names == ["patchcore", "efficientad", "anomalydino"]


def test_final_train_config_points_to_benchmark() -> None:
    config = load_final_train_config("configs/modern/final_train.yaml")
    assert config.benchmark_config.exists()
