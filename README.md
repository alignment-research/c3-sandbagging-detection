# Cross-Context Consistency (C³) for Black-Box Sandbagging Detection

Code for *Same Question, Different Lies: Cross-Context Consistency (C³) for Black-Box Sandbagging Detection* (ICML 2026).

C³ detects sandbagging (deliberate underperformance) with black-box access only. It asks a model the same question under several paraphrases, each in a fresh conversation. It then embeds the model's explanations and measures how much they disagree. A model that truly lacks the capability fails in a consistent way. A capable model that feigns incompetence must invent a new failure for each context, so its answers are less consistent.

The repository also contains the baselines from the paper: trusted monitoring, lie detectors and behavioral probes. It also contains the model organisms: prompted, password-locked and discovered-incentive sandbaggers.

## Install

The code needs Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/alignment-research/c3-sandbagging-detection.git
cd c3-sandbagging-detection
uv sync
```

`uv.lock` pins dependencies as of the camera-ready date (May 2026). Put API keys in a `.env` file in the repository root:

```bash
OPENAI_API_KEY=...       # models under test, paraphraser, text-embedding-3-small
ANTHROPIC_API_KEY=...    # trusted monitor (Claude Haiku) and Claude models
GOOGLE_API_KEY=...       # Gemini models (discovered-incentive / ProtocolQA runs)
```

USACO tasks run model code in a sandbox (Docker, or Modal for the `*-modal` configs); see `src/detection_game/tasks/docker_setup.md`. Password-locked experiments need a GPU and the `local-inference` extra (`uv sync --extra local-inference`).

## Data

No data is bundled. Datasets are fetched on first use, except GPQA.

| Dataset | Source | How to get it |
|---|---|---|
| GPQA Diamond | OpenAI simple-evals | `mkdir -p data && curl -L https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv -o data/gpqa_diamond.csv` |
| MATH (Level 5) | `EleutherAI/hendrycks_math` (Hugging Face) | automatic |
| USACO | USACOBench (Shi et al., 2024) | automatic, cached under the `inspect_evals` user cache dir |
| WMDP | `cais/wmdp` (Hugging Face) | automatic |
| GSM8K | `openai/gsm8k` (Hugging Face) | automatic |
| ProtocolQA | `futurehouse/lab-bench` (Hugging Face) | automatic |

Password-locked models: `scripts/setup_local_models.sh` downloads `redwoodresearch/math_pwd_lock_deepseek_math7b_on_weak_pythia1b` (MATH and GSM8K). `scripts/merge_wmdp_peft_model.py` merges `FelixHofstaetter/mistral-7b-sandbagging-new` (WMDP). The model configs expect them under `~/models/`.

No model outputs or score files are included, so reproducing a number means rerunning its experiment. The paper's figures were drawn from these scores in the paper's own repository.

## How the pipeline works

Every experiment has the same three stages. Experiments are [Hydra](https://hydra.cc/) configs in `src/detection_game/configs/`.

1. **Generate trajectories.** `cli.run_sweep` runs the benign and sandbagging models on a dataset and writes Inspect `.eval` logs to `logs/runs/`.
2. **Score each question.** A classifier in `src/detection_game/classifiers/` turns the logs into one score per question. It writes a CSV.
3. **Aggregate and threshold.** `classifiers.run_multisample_bootstrap_cv` bootstraps k questions per model, calibrates the threshold at 5% FPR on benign models, and reports recall.

Add `trajectory.eval.limit=5` to any `run_sweep` command for a cheap smoke test.

## Reproducing the main results

The commands below use GPQA and GPT-5-nano (the same-model setting). MATH and USACO use the same steps with their own configs (`math-embedding-entropy-5nano*`, `usaco-embedding-entropy-5nano`, and so on).

### C³

```bash
# 1. Paraphrased trajectories: sandbagging prompts and the benign baseline (test split)
uv run python -m src.detection_game.cli.run_sweep \
  --config-name=gpqa-embedding-entropy-5nano-test --multirun
uv run python -m src.detection_game.cli.run_sweep \
  --config-name=gpqa-embedding-entropy-5nano-benign-test --multirun

# 2. Per-question C³ score: mean pairwise cosine distance of the explanation embeddings.
#    The scorer reads one directory, so link both runs' logs into one directory first.
mkdir -p logs/runs/gpqa-c3-5nano-test-combined
ln -s "$PWD"/logs/runs/gpqa-embedding-entropy-5nano-test-*/*.eval \
  "$PWD"/logs/runs/gpqa-embedding-entropy-5nano-benign-test-*/*.eval \
  logs/runs/gpqa-c3-5nano-test-combined/
uv run python -m src.detection_game.classifiers.run_consistency_classifier \
  classifier.log_dir=logs/runs/gpqa-c3-5nano-test-combined \
  classifier.message_selection=last \
  classifier.grouping_mode=paraphrasing \
  classifier.strategy_filter=null \
  'classifier.metrics=[mean_pairwise_cosine_distance]' \
  classifier.embedding_model=text-embedding-3-small \
  classifier.output.csv=out/gpqa_c3.csv

# 3. Recall at 5% FPR with k=10 questions per model
uv run python -m src.detection_game.classifiers.run_multisample_bootstrap_cv \
  classifier.input_csv=out/gpqa_c3.csv \
  classifier.k_samples=10 \
  classifier.aggregation=mean \
  classifier.compare_mode=all-pairs
```

For categorical entropy on multiple-choice answers (GPQA), use `'classifier.metrics=[categorical_entropy]'` with `classifier.message_selection=0`.

### Baselines

| Method | Trajectory configs (GPQA; MATH and USACO analogues exist) | Scorer |
|---|---|---|
| Trusted monitoring | `gpqa-trusted-monitor-5nano-test`, then the monitor pass `cli.run_judge --config-name=gpqa-trusted-monitor judge.dataset.path=logs/runs/<trajectory run>` (MATH: `math-tm-judge-5nano-test`) | `classifiers.run_trusted_monitor` |
| Lie detectors | `gpqa-lie-detector-5nano-test` | `classifiers.run_lie_detector` (pretrained detectors in `src/detection_game/utils/lie_detectors/models/`) |
| Behavioral probes | `gpqa-behavioral-probes-5nano-test` | `classifiers.run_behavioral_probe` |

Each scorer writes a CSV via `classifier.output.csv=...`. Step 3 above then aggregates that CSV. `scripts/run_honest_wrong_analysis.sh` shows complete invocations of all three baselines.

### Other experiments

| Paper section | Configs and scripts |
|---|---|
| Capable sandbaggers vs weaker benign models (cross-capability) | `3mfpr-*` configs, `classifiers.run_3model_fpr` |
| Password-locked sandbaggers | `{gsm8k,math,wmdp}-password-locked-*`, `scripts/run_gsm8k_password_locked_experiments.sh` |
| Discovered-incentive sandbaggers (GPT-5.4, Gemini 3.1 Pro) | `*-goal-conflict-*-gpt54`, `*-meinke-*`, `protocolqa-open-c3-gemini-native-tc*` |
| Ensembling with trusted monitoring | `classifiers.run_ensemble_detector`, `scripts/compute_*_ensemble.py` |
| Lie detector model-identity confound | `scripts/analyze_lie_detector_confound.py` |
| Number of questions k | `scripts/plot_multisample_recall_vs_k.py` |
| Self-calibration without a benign reference | `src/detection_game/analysis/robust_selfcal.py` |

Sandbagging system prompts are in `src/detection_game/configs/system_prompt/` (index: `CATALOGUE.yaml`). Paraphrase strategies are in `src/detection_game/configs/paraphraser/`.

## Tests

```bash
uv run pytest -m "not slow and not integration"
```

## Citation

```bibtex
@inproceedings{lin2026c3,
  title     = {Same Question, Different Lies: Cross-Context Consistency ({C}$^3$) for Black-Box Sandbagging Detection},
  author    = {Lin, Yulong and Bernabeu-P{\'e}rez, Pablo and Arnav, Benjamin and Wells, Lennie and Phuong, Mary},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
