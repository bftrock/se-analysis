# se-analysis

This project is used to perform various analyses required of Solutions Engineers doing data QC.

The environment is managed with `uv`, a fast package and environment manager. To get started:

1. Install `uv`: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
1. Create the environment specified in `pyproject.toml`: `uv sync`

You can find template Jupyter Notebooks in the `notebooks` directory. Shared helper code lives in the `se_analysis` package under `src`; it's installed in editable mode by `uv sync`, so notebooks can `import se_analysis` from anywhere.
