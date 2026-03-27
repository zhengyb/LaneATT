# 超参调优化记录


## 1. Base

```
============================================================
  Comparing Phase 0 results (pick best)
============================================================
$ python3 scripts/compare_results.py --pick-best ablation_bs8_lr0001 ablation_bs8_lr0003 ablation_bs8_lr001

Phase 0 lr search — pick best (max_epoch=15)

experiment                                  lr   loss_avg       status  selected
--------------------------------------------------------------------------------
ablation_bs8_lr0001                   0.000100   26.60629  oscillation <-- lr_base
ablation_bs8_lr0003                   0.000300   27.91193  oscillation          
ablation_bs8_lr001                    0.001000   28.87179  oscillation          

  lr_base = 0.0001

  Derived lr values for subsequent phases:
    B/D (x2)      = 0.0002
    C/E (xsqrt2)  = 0.000141
    F/H (x4)      = 0.0004
    G   (x2)      = 0.0002

>>> Phase 0 winner: ablation_bs8_lr0001
>>> Next step: run scripts/generate_configs.py --base <base.yml> --lr-base <lr>
>>>            then: python3 scripts/run_ablation.py phase1 --lr-base <lr>
```