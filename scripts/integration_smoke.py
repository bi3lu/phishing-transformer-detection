"""Offline CPU integration check on a tiny transformer; never research results."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    root = Path(tempfile.mkdtemp(prefix="phishing-integration-"))
    print(f"Isolated integration workspace: {root}", flush=True)
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PHISHING_TRAIN_DEVICE": "cpu",
        "PHISHING_RUN_ID": "integration_smoke",
        "PHISHING_SEED": "42",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(root / "mpl-cache"),
        "XDG_CACHE_HOME": str(root / "cache"),
    }
    os.environ.update(environment)

    import yaml
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    for folder in ("src", "docs", "data/raw"):
        shutil.copytree(repository / folder, root / folder, ignore=shutil.ignore_patterns("__pycache__"))

    for name in ("main.py", "uv.lock"):
        shutil.copy2(repository / name, root / name)

    model_path = root / "tiny-model"
    vocabulary = {word: i for i, word in enumerate(("[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "SMS", "test"))}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(model_path)
    BertForSequenceClassification(  # type: ignore[no-untyped-call]
        BertConfig(  # type: ignore[no-untyped-call]
            vocab_size=len(vocabulary),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    ).save_pretrained(model_path)
    config = yaml.safe_load((repository / "params.yaml").read_text())
    config["experiments"] = [
        {**config["experiments"][0], "name": "tiny-transformer", "model_name": str(model_path), "epochs": 1}
    ]
    config["research_protocol"]["cross_validation"]["bootstrap_resamples"] = 100
    config["research_protocol"]["final_test_uncertainty"]["resamples"] = 100
    (root / "params.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    def run(arguments: list[str], log_name: str) -> None:
        with (root / log_name).open("w") as log:
            subprocess.run([sys.executable, *arguments], cwd=root, env=environment, stdout=log, stderr=log, check=True)

    run(["main.py"], "pipeline.log")
    tracked = {
        path: path.stat().st_mtime_ns
        for path in (root / "results").rglob("*")
        if path.name in {"model.safetensors", "pipeline.joblib", "completed.json"}
    }
    run(["main.py", "--skip", "preprocess", "split"], "resume.log")

    if not tracked or any(path.stat().st_mtime_ns != stamp for path, stamp in tracked.items()):
        raise AssertionError("Resume rewrote completed training artifacts")

    run(
        [
            "-c",
            "from src.config import SAVED_MODELS_DIR; "
            "from src.evaluation.explainer import PhishingExplainer; "
            "import numpy as np; "
            "x=PhishingExplainer(str(SAVED_MODELS_DIR/'tiny-transformer')); "
            "e=x.get_explanation('test SMS'); d=x.decision('test SMS'); "
            "assert np.isclose(e.base_values[0,1]+e.values[0,:,1].sum(),d['probability'],atol=1e-5); "
            "assert d['prediction']==int(d['probability']>=d['threshold']); x.close()",
        ],
        "shap.log",
    )
    run(["-m", "src.data.archive", "dataset.tar.gz"], "archive.log")
    run(["-m", "src.data.archive", "dataset.tar.gz", "--verify"], "archive-verify.log")
    result = {"passed": True, "unchanged_training_artifacts": len(tracked), "research_results": False}
    (root / "integration_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
