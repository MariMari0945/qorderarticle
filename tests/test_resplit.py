import pandas as pd

from qorder import run


def test_resplit_gives_each_seed_its_own_split(tmp_path):
    cfg = {"name": "t_resplit", "dataset": {"name": "digits", "classes": [3, 8], "n_components": 8},
           "model": {"kind": "shallow_reference", "n_qubits": 8, "observable": "local",
                     "learning_rate": 0.1, "max_steps": 5},
           "grid": {"n_layers": [1]}, "orderings": ["heuristic", "default"], "seeds": [0, 1],
           "resplit": True}
    df = run.run_training(cfg, tmp_path, quick=True)
    assert sorted(df.split_seed.unique()) == [0, 1]
    assert (df.groupby("split_seed").size() == 2).all()
    assert (tmp_path / "dataset_info_split0.json").exists() and (tmp_path / "dataset_info_split1.json").exists()
    # rerun skips everything already trained
    df2 = run.run_training(cfg, tmp_path, quick=True)
    assert len(df2) == len(df)


def test_fixed_split_unchanged(tmp_path):
    cfg = {"name": "t_fixed", "dataset": {"name": "digits", "classes": [3, 8], "n_components": 8},
           "model": {"kind": "shallow_reference", "n_qubits": 8, "observable": "local",
                     "learning_rate": 0.1, "max_steps": 5},
           "grid": {"n_layers": [1]}, "orderings": ["default"], "seeds": [0, 1]}
    df = run.run_training(cfg, tmp_path, quick=True)
    assert (df.split_seed == -1).all() and (tmp_path / "dataset_info.json").exists()
