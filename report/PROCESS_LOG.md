# MP1 process log

This file is the human-readable experiment diary. Machine-readable results are
kept in `RUN_LOG.csv` and each run's `metrics.json` / evaluation JSON.

## 2026-09-22 — package and environment audit

- Read `GUIDE.md`, the code `README.md`, source, tests, manifest, and baseline
  configuration.
- Confirmed fixed protocol: `7506-mp1-wt2-v2`; vocabulary 2048; context 256;
  independent causal evaluation windows.
- Created a Python 3.12 project-local `.venv` and installed the required CPU
  PyTorch build for reproducible ranked evaluation.
- Added VS Code workspace settings, tasks, debug configurations, and developer
  instructions without changing the fixed evaluator or data.
- Hardware detected: NVIDIA GeForce GTX 1660-series GPU with 6 GiB VRAM. The
  official baseline and ranked score are nevertheless measured on CPU FP32.
- Noted and fixed a packaging inconsistency: the original README link pointed
  to `../guide/GUIDE.md`, while the extracted package placed `GUIDE.md` at the
  package root. The submission repository now keeps `GUIDE.md` at its root.

## Baseline reproduction

Status: complete.

- Environment: Windows 11 Pro 10.0.26100; AMD Ryzen 7 4800HS (8 cores / 16
  logical processors); 16 GB system RAM; Python 3.12.13; PyTorch 2.7.1+cpu.
- Contract tests: 5/5 passed.
- Smoke run: 10 updates completed; checkpoint reload and full-test scorer passed.
- Full command: `python train.py --implementation model --device cpu
  --precision fp32 --threads 4 --seed 17 --run-dir runs/baseline-s17`.
- Training: 1,200 updates, batch size 32, 9,830,400 processed targets, 769.992 s.
- Model: 1,088,256 parameters; final sampled training loss 4.316905.
- Validation: 2.071083 BPB; 11.785 s.
- Test command: `python evaluate.py --checkpoint
  runs/baseline-s17/checkpoint.pt --device cpu --precision fp32 --threads 4
  --split test`.
- Full test: 2.101260 BPB; token perplexity 80.8471; 428,405 targets; 13.459 s.
- Checkpoint SHA-256:
  `7d869d6637d684bf4d11a9c91a9c7a3fda38191cc6849054665a24a649ea6d3f`.
- Baseline inference assets (checkpoint plus implementation module): 4.171 MiB.
- Windows process-tree peak working set during a repeated scoring run: 1.5384
  GiB. The repeated score was identical and took 12.060 s. This is an OS working
  set measurement, not the CUDA-only memory field emitted by the evaluator.
- The first attempted memory measurement saw only the `.venv` launcher process
  (0.0039 GiB) and is explicitly discarded as invalid; the process-tree result
  above includes its spawned base-Python process.
- Result matches the staff's approximate 2.10 test BPB reference.

## Candidate development

### 2026-09-26 — parameter-matched SwiGLU

- Diagnosed the baseline GELU MLP as a single nonlinear feature-transformation
  path without a learned multiplicative gate.
- Implemented `student_model.py`: the supplied attention, normalization,
  positional embeddings, residual layout, initialization and tied output head
  are preserved; only the token-wise MLP becomes SwiGLU.
- Used hidden width 341 to parameter-match the baseline: 1,088,424 versus
  1,088,256 parameters (168 extra, 0.015%).
- Contract tests: 5/5 passed. A 10-step smoke training/checkpoint run passed.
- Fair run command: `python train.py --implementation student --config
  configs/swiglu_128.json --device cpu --precision fp32 --threads 4 --seed 17
  --steps 1200 --batch-size 32 --run-dir runs/swiglu128-fair-s17`.
- Fair run: 9,830,400 targets; 724.216 s training; validation 2.010852 BPB in
  11.617 s; checkpoint SHA-256
  `a555a374fec037e3bfbba6b2920287bfb67ff77fa374d2d66e9014ac18b83c94`.
- Versus baseline, validation BPB improved by 0.060232 (2.91% relative), with
  essentially identical parameter count. This validation result selected
  SwiGLU for the final candidate.
- Started a width-192, 6-head, 4-block SwiGLU candidate for 3,600 steps
  (29,491,200 targets). This is an additional quality/cost trade-off experiment,
  not the controlled mechanism ablation.

### 2026-09-26 — final frozen predictor

- Final configuration: width 192, six heads, four blocks, SwiGLU hidden width
  512; 2,223,232 parameters.
- Training: 3,600 updates, batch size 32, 29,491,200 processed targets,
  3,483.082 s. Validation: 1.710366 BPB in 16.554 s.
- Validation was substantially below both the matched SwiGLU run (2.010852) and
  baseline (2.071083), so this checkpoint was frozen before full-test scoring.
- Frozen checkpoint SHA-256:
  `373b884e4daec5c99ad2e0645106ead62580b68c2b07a0eb2220e3c7c1c87dcc`.
- CPU FP32 full test: 1.738670 BPB, token perplexity 37.8862, 428,405 targets,
  19.928 s. This is 0.362590 BPB / 17.26% below the baseline test score.
- Repeated resource-instrumented scoring reproduced 1.738670 BPB in 20.055 s.
  Windows process-tree peak working set was 1.5333 GiB.
- Inference assets counted conservatively as checkpoint, `student.py`,
  `student_model.py`, and configuration: 8,917,166 bytes / 8.5041 MiB.
- Constraint checks: scoring time 1.48x baseline (<5x); RAM 1.5333 GiB
  (<4 GiB); inference assets 8.5041 MiB (<64 MiB); contract tests 5/5 passed.

## 2026-09-28 - capacity-scaled dropout candidate

- Added configurable dropout to the existing SwiGLU block while preserving the
  causal attention, normalization, tokenizer and evaluator contracts. The new
  configuration is `configs/swiglu_256x6_dropout.json`: width 256, eight heads,
  six blocks, hidden width 688 and dropout 0.1 (5,355,584 parameters).
- Trained for 12,000 updates with seed 17 and batch size 32, processing
  98,304,000 targets. Training used local CUDA FP32 only as an acceleration
  device; this does not define the submitted score. GPU training took
  2,207.836 s and used 1.3148 GB peak allocated memory.
- Validation history improved from 1.759939 at step 2,000 to 1.567424 at step
  12,000. The method was frozen before test evaluation.
- The authoritative full-test command used the original CPU environment:
  `python evaluate.py --checkpoint runs/swiglu256x6-dropout10-final12000-s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test`.
  It produced **1.589530620 BPB** in 42.991 s, with 428,405 targets and
  1,292,013 UTF-8 bytes. Checkpoint SHA-256:
  `90163aed08c92f1affce1830907e63a62ef0b01ec32327c82a71088f8c747802`.
- The CPU timing is 3.19x the locally reproduced baseline (13.459 s), below
  the 5x limit. The checkpoint is 21.45 MB; the evaluator reports zero CUDA
  allocation because this final measurement is CPU-only. The earlier Windows
  process-tree RAM procedure should be rerun before a formal submission if the
  course requests an OS-level peak-RAM number for this larger candidate.
## 26 September 2026 - report revision and submission hygiene

- Replaced the stale matched-model test placeholder with `Not evaluated
  (validation-only ablation)`; no test score was invented or added.
- Expanded the Word report with a baseline-component analysis, the reason for
  selecting SwiGLU, clearer separation of controlled and scaled experiments,
  additional interpretation of the validation/test results, and a plain-language
  explanation of checkpoint SHA-256 fingerprints.
- Verified the cited SwiGLU and WikiText papers against their original arXiv
  records and added numbered citations in the report body.
- Added the required AI-assistance and reused-work disclosure to the code README.
- Renamed the staff checksum list to `ORIGINAL_PACKAGE_MANIFEST.json` so it is
  not mistaken for a manifest of the modified submission.
- Changed the legacy report builders to write separate `*_legacy_rebuild` files,
  protecting the student-edited and revised reports from accidental overwrite.
