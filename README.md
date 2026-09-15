# se-analysis

This project is used to perform various analyses required of Solutions Engineers doing data QC.

The environment is managed with `uv`, a fast package and environment manager. To get started:

1. Install VS Code. Download and run the installer. You don't need admin privs to do this; you can run it under your profile if you prefer.
2. Install these VS Code extensions:
   - Claude Code for VS Code
   - Data Wrangler
   - Jupyter
   - Jupyter Cell Tags
   - Jupyter Key Map
   - Jupyter Notebook Renderers
   - Jupyter Slide Show
   - Pylance
   - Python
   - Python Debugger
   - Python Environments
3. Install `uv`. Run this with PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`. If given the option, make sure `uv` is in your PATH so you can run commands with it.
4. Clone this repo to your local system in the location of your choice.
5. Create the environment specified in `pyproject.toml`: `uv sync`
6. Create a .env file in the `se-analysis` directory by saving as .env.example and assigning the 3 variables in that file. This allows you to retrieve data from NRG Cloud, and specifies your project directory, which is where you'll do your work and save data files.

Here's how you might organize your project directory:

```text
Projects/                                   <- the directory named in your .env
├── .env
├── SRA/                                    <- product line / analysis type
│   ├── Pecos-Flats_TX/                     <- one directory per site, Site-Name_ST
│   │   ├── Process_SymPRO_SRA_Data_Pecos-Flats_TX_2026-08.ipynb
│   │   ├── Process_SymPRO_SRA_Data_Pecos-Flats_TX_2026-09-08.ipynb
│   │   └── data/                           <- raw exports, left unmodified
│   │       ├── 000680_Apex_Clean_Energy_meas_2026.08.05-2026.08.05.txt
│   │       └── 000680_Apex_Clean_Energy_meas_2026.08.06-2026.08.06.txt
│   └── Two-Rivers_NY/
│       ├── Process_SymPRO_SRA_Data_Two-Rivers_NY_2026-09-08.ipynb
│       └── data/
│           └── 003218_Boralex_meas_2026.08.01-2026.08.01.txt
├── SRM/
│   ├── Jackson_MI/                         <- single-logger site: notebooks + data/
│   │   ├── Process_LOGR_Diag_Files_Jackson_MI.ipynb
│   │   ├── Process_LOGR_Log_Files_Jackson_MI.ipynb
│   │   ├── Process_LOGR_Meas_Files_Jackson_MI.ipynb
│   │   ├── Jackson_MI_Log-Files_filtered_export.csv    <- analysis output
│   │   └── data/
│   │       ├── 20260904_0000_002682_001289.diag
│   │       ├── 20260904_0000_002682_001289.log
│   │       └── 20260904_0000_002682_001289_statistical.dat
│   └── Pierce-County_NE/                   <- multi-tower site: one directory per MET
│       ├── Validate_ProtoNode_File_Pierce-County_NE_MET1.ipynb
│       ├── MET1/
│       │   └── Pierce_MET1_MDC_TCP_192_168_2_8_ProtoNode_2026-07-10_15_22_04.txt
│       └── MET2/
│           └── Pierce_MET2_MDC_TCP_192_168_22_8_ProtoNode_2026-07-11_08_40_04.txt
└── SR300-Issue/                            <- cross-site investigation
    ├── All-Log-Messages.csv
    ├── MB-Faults-Export.csv
    ├── MET01/
    │   ├── 20260728_0000_002544_002561.diag
    │   ├── 20260728_0000_002544_002561.log
    │   └── 20260728_0000_002544_002561_statistical.dat
    └── MET02/
        └── ...
```

A few conventions worth keeping:

- Group by product line or investigation first, then by site (`Site-Name_ST`), then by tower (`MET1`, `MET2`, ...) if the site has more than one.
- Keep raw logger and ProtoNode exports in a `data/` subdirectory (or in the per-`MET` directory) and leave them unmodified; save analysis outputs alongside the notebook.
- Avoid spaces in directory and file names so paths stay easy to type and to script against.

You can find template Jupyter Notebooks in the `notebooks` directory. Shared helper code lives in the `se_analysis` package under `src`; it's installed in editable mode by `uv sync`, so notebooks can `import se_analysis` from anywhere.

You can use VS Code to run Jupyter Notebooks, or you can run Jupyter Lab from the `se-analysis` Python environment, which ensures that the shared code is available in Jupyter Lab.

## Using VS Code to run a Jupyter Notebook

With VS Code, you don't start a Jupyter Lab server. Instead, you open the notebook file, then select the kernel to run it. The advantages:
- You're not bound to a single root directory as you are in Jupyter Lab.
- It has auto-complete to help you write code.
- You can use the **Claude Code for VS Code** extension with the NRG Claude subscription to create and modify files on your disk.
- Runs faster than Jupyter Lab, especially when rendering many plots.
- The **Data Wrangler** VS code extension handles DataFrames really nicely. It paginates the rows and allows you to filter and sort rows right in the output cell. Much more useful than just printing a DataFrame.

Here's how to use it:

1. Start with one of the template notebooks and Save As to your project directory.
2. Select the kernel in the upper right. You should see something like `Python (se-analysis)` in the drop-down.

## Running Jupyter Lab

Alternatively, you can run Jupyter Lab from the `se-analysis` Python environment. The upside is a familiar browser-based interface, but the downside is you can only have a single root directory. If you want to start with a template notebook in the `notebooks` directory of the repo, you'll have to copy the notebook(s) to your project directory so you can see it in the Jupyter Lab environment.

1. Open a terminal (cmd or PowerShell) and change into the `se-analysis` directory.
2. Start the Jupyter Lab server: `uv run jupyter lab --ServerApp.root_dir=[your project directory]`