# Scripts

Utility and evaluation scripts for `kornia-slam`.

## Scripts

- **`start_time_eval.py`**: Evaluation harness to measure initialisation sensitivity, time-to-init, scale error, and trajectory accuracy across start-time offsets (issue #168).
  ```bash
  # Run self-contained synthetic smoke test:
  python scripts/start_time_eval.py --smoke-test --output-dir eval_output/smoke_test

  # Run on an EuRoC sequence:
  python scripts/start_time_eval.py \
      --dataset-path data/euroc/MH_05_difficult \
      --sequence-name MH_05_difficult \
      --config configs/euroc.ron \
      --start-offsets 0,100,200,300,400,600 \
      --output-dir eval_output/mh05
  ```
- **`check_version_pins.py`**: Checks that crate versions and dependencies match workspace release pins.
- **`release_rust.sh`**: Pre-release packaging and checks.
