import json

import numpy as np
from PIL import Image

from sfvmosaic.eval_suite import add_noise, run_eval_suite


def test_add_noise_keeps_shape_and_dtype():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    noisy = add_noise(image, "gaussian", 2.0, seed=1)
    assert noisy.shape == image.shape
    assert noisy.dtype == np.uint8


def test_eval_suite_writes_feasible_outputs(tmp_path):
    rng = np.random.default_rng(31)
    secret = rng.integers(80, 180, (32, 32, 3), dtype=np.uint8)
    target = rng.integers(80, 180, (32, 32, 3), dtype=np.uint8)
    secret_path = tmp_path / "secret.png"
    target_path = tmp_path / "target.png"
    output_path = tmp_path / "eval"
    Image.fromarray(secret, "RGB").save(secret_path)
    Image.fromarray(target, "RGB").save(target_path)

    summary = run_eval_suite(
        str(secret_path),
        str(target_path),
        str(output_path),
        [16],
        variants=[
            {
                "name": "paper_baseline",
                "mode": "paper",
                "match_strategy": "avg_std",
                "q_strategy": "paper",
                "use_rotation": True,
                "residual_strategy": "default",
            },
            {
                "name": "ablate_rotation",
                "mode": "paper",
                "match_strategy": "avg_std",
                "q_strategy": "paper",
                "use_rotation": False,
                "residual_strategy": "default",
            },
        ],
        noises=[{"name": "none", "kind": "none", "level": 0.0}],
        embedded=False,
    )

    assert summary["quality_rows"] == 2
    assert summary["noise_rows"] == 2
    assert (output_path / "quality.csv").exists()
    assert (output_path / "noise.csv").exists()
    assert json.loads((output_path / "summary.json").read_text())["quality_rows"] == 2
