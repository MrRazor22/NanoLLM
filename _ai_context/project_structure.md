# Project Structure

```text
Piqo/
├── benchmark/
│   ├── competitors/
│   │   ├── gliner_decide.py
│   │   ├── jev.py
│   │   ├── kev.py
│   │   ├── laya.py
│   │   └── verdict.py
│   ├── policies/
│   │   ├── ledger.py
│   │   ├── reporter.py
│   │   └── scorer.py
│   ├── suites/
│   │   ├── manifests/
│   │   ├── jevbench.py
│   │   ├── kev_transfer.py
│   │   ├── laya_typed.py
│   │   └── suite.py
│   ├── __main__.py
│   ├── benchmark.py
│   └── registry.py
├── data/
│   ├── corpus/
│   ├── curation/
│   │   ├── budget_split.py
│   │   └── ngram_leakage.py
│   ├── raw/
│   ├── sources/
│   │   ├── anli_source.py
│   │   ├── art_source.py
│   │   ├── copa_source.py
│   │   ├── decisions_source.py
│   │   ├── glaive_source.py
│   │   ├── hellaswag_source.py
│   │   ├── huggingface_source.py
│   │   ├── medqa_source.py
│   │   ├── multichoice_source.py
│   │   ├── reclor_source.py
│   │   ├── sciq_source.py
│   │   └── source.py
│   ├── __main__.py
│   ├── case.py
│   ├── dataset.py
│   ├── registry.py
│   └── storage_policy.py
├── inference/
│   ├── __main__.py
│   ├── decision_engine.py
│   ├── decision_resolver.py
│   ├── profiling_layer.py
│   └── questions.py
├── model/
│   ├── policies/
│   │   ├── calibrated_loss.py
│   │   ├── head.py
│   │   ├── layout.py
│   │   ├── optimizer.py
│   │   └── weight_cache.py
│   ├── weights/
│   ├── device.py
│   ├── token_layout.py
│   ├── torch_model.py
│   └── weights.py
├── training/
│   ├── policies/
│   │   ├── batch.py
│   │   └── schedule.py
│   ├── telemetry/
│   │   ├── checkpointing_layer.py
│   │   └── metrics_layer.py
│   ├── __main__.py
│   ├── epoch_trainer.py
│   └── stats.py
└── piqo.py
```
