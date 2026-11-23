"""Publish the deployment bundle to a Hugging Face model repo.

    uv run --with huggingface_hub python scripts/publish_hf.py --repo <user>/penumbra-nids

Uploads what the API needs at runtime and git does not carry: the registry champions (each verified
against its manifest before it is copied, exactly as deploy_azure.py does), the held-out sample
pools, the evaluation reports and the ATT&CK corpus, plus a model card. The free Render instance
fetches from here at build time (scripts/fetch_bundle.py, deploy/render.Dockerfile).

Needs a write token: `hf auth login` once, or HF_TOKEN in the environment. Never prints it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_azure import ROOT, stage_models  # noqa: E402

CARD = """---
license: mit
library_name: sklearn
tags:
  - tabular-classification
  - intrusion-detection
  - network-security
  - anomaly-detection
  - cybersecurity
---

# Penumbra: network intrusion detectors

The three registry champions behind [Penumbra](https://github.com/{gh}), an ML network intrusion
detector that alerts a SOC and never blocks traffic: a 300-tree random forest plus a benign-only
novelty head, each thresholded at a target false-positive rate on held-out benign rows.

| path | trained on |
|---|---|
| `models/unsw/` | UNSW-NB15 |
| `models/nslkdd/` | NSL-KDD |
| `models/cicids/` | CICIDS2017, Monday to Wednesday (Thursday and Friday held out) |

Each `detector.joblib` is a pickled `penumbra.models.detector.PenumbraDetector`; load it with
`PenumbraDetector.load(<dir>)` from the repository, at the versions pinned in its `uv.lock`.
Pickles execute code when loaded: only load files you have verified. `metadata.json` carries the
registry version, features and thresholds.

Also here: `samples/` (held-out test flows the live demo draws from), `reports/` (the evaluation
reports the console renders) and `data/interim/attack_techniques.jsonl` (the MITRE ATT&CK text the
triage copilot retrieves from; ATT&CK is (c) The MITRE Corporation, used under its terms of use).

Evaluation, method and known weaknesses: the repository's `docs/EVALUATION.md` and `docs/MODEL_CARD.md`.
"""


def main(repo: str, github: str, private: bool) -> None:
    from huggingface_hub import HfApi

    from penumbra.config import settings
    from penumbra.rag.corpus import CORPUS_FILENAME

    api = HfApi()
    stage = Path(tempfile.mkdtemp(prefix="penumbra-hf-"))
    try:
        stage_models(stage)
        out = stage / "repo"
        shutil.move(stage / "deploy" / "models", out / "models")
        shutil.move(stage / "deploy" / "samples", out / "samples")
        (out / "reports").mkdir()
        for report in (ROOT / "artifacts" / "reports").glob("*.json"):
            shutil.copy2(report, out / "reports" / report.name)
        corpus = settings().interim_dir / CORPUS_FILENAME
        if corpus.exists():
            (out / "data" / "interim").mkdir(parents=True)
            shutil.copy2(corpus, out / "data" / "interim" / CORPUS_FILENAME)
        else:
            print("  no ATT&CK corpus found; triage notes will say so (penumbra copilot build)")
        (out / "README.md").write_text(CARD.replace("{gh}", github), encoding="utf-8")

        api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo,
            repo_type="model",
            folder_path=out,
            commit_message="Deployment bundle: champions, sample pools, reports, ATT&CK corpus",
            delete_patterns=["models/**", "samples/**", "reports/**", "data/**"],
        )
        print(f"https://huggingface.co/{repo}  ({commit.oid[:10]})")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="<hf-user>/penumbra-nids")
    parser.add_argument("--github", default="dhruvv1402/penumbra-nids")
    parser.add_argument(
        "--private", action="store_true", help="the Render build fetches anonymously; keep public"
    )
    args = parser.parse_args()
    main(args.repo, args.github, args.private)
