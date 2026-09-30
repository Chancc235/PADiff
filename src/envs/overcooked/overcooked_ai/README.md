# Overcooked-AI dependency

This directory contains the Python Overcooked-AI environment from the
[Center for Human-Compatible AI project](https://github.com/HumanCompatibleAI/overcooked_ai),
used by PADiff for training and evaluation. Its license is in [LICENSE](LICENSE).

The browser demo and JavaScript implementation are omitted. The `assets/`
directory retains the sprites used by PADiff's Python renderer. The same
dependency is linked from `PADiff/envs/overcooked/overcooked_ai` so both
training stages use one copy.
