# Terminal Shell Configuration Plan

## Overview

Configure Bob IDE's integrated terminal to:
1. Use **zsh** instead of bash as the default shell
2. Automatically activate the Python environment on terminal open:
   - If `.venv/` exists → `source .venv/bin/activate`
   - Else if `.conda_config` exists → extract the env path and run `conda activate <path>`

This is workspace-local (developer machine) configuration. The approach uses:
- **`.vscode/settings.json`** — tells the IDE terminal to use zsh and to point `ZDOTDIR` to a project-local directory containing our custom init script
- **`.vscode/.zshrc`** — the custom zsh startup script containing the conditional activation logic

`.vscode/` is already in `.gitignore` so these files won't be committed.

---

## Sub-Tasks

---

### Sub-Task 1: Create `.vscode/settings.json` — Terminal Profile for zsh

**Intent**
Configure the IDE's integrated terminal to launch zsh by default. Use `ZDOTDIR` to redirect zsh's startup file lookup to our project-local `.vscode/` directory, where we'll place a custom `.zshrc`.

**Why `ZDOTDIR`?**
When zsh starts, it reads `$ZDOTDIR/.zshrc` (or `~/.zshrc` if `ZDOTDIR` is unset). By setting `ZDOTDIR` to the workspace `.vscode/` folder via the terminal env, we can intercept startup without touching the user's global `~/.zshrc`.

**Expected Outcomes**
- Opening a new terminal in Bob IDE launches zsh (not bash)
- The terminal env has `ZDOTDIR` set to the absolute path of `.vscode/` in this workspace
- The user's personal `~/.zshrc` is sourced first (for their PATH, aliases, prompt, etc.), then the project activation runs

**Todo List**
1. Create `.vscode/settings.json` with the following settings:
   - `terminal.integrated.defaultProfile.linux` → `"zsh"`
   - `terminal.integrated.profiles.linux` → define a `"zsh"` profile using the zsh binary (e.g., `/bin/zsh` or `$(which zsh)`)
   - `terminal.integrated.env.linux` → set `ZDOTDIR` to `${workspaceFolder}/.vscode`

**Relevant Context**
- Workspace root: `/dccstor/cardiac/QBioCode`
- `.vscode/` is in `.gitignore` (line 30) — these files are local-only, correct behavior
- No existing `.vscode/settings.json` — create fresh

**Status:** [x] done

---

### Sub-Task 2: Create `.vscode/.zshrc` — Conditional Environment Activation Script

**Intent**
Write the custom zsh startup file that:
1. Sources the user's real `~/.zshrc` first (so prompt, PATH, aliases still work)
2. Then applies conditional project-environment activation logic

**Activation Logic**
```
PROJECT_DIR = the workspace folder
if PROJECT_DIR/.venv exists → source PROJECT_DIR/.venv/bin/activate
elif PROJECT_DIR/.conda_config exists → read path from it → conda activate <that path>
```

**`.conda_config` format assumption**: a plain-text file whose first non-empty, non-comment line is the path to the conda environment (e.g., `/dccstor/cardiac/miniforge3/envs/myenv`). This needs to be confirmed or adjusted during implementation.

**Expected Outcomes**
- A new terminal opens in zsh
- If `.venv/` exists: the uv venv is activated (prompt shows `(QBioCode)`)
- If `.venv/` doesn't exist and `.conda_config` exists: conda activates the extracted path
- If neither: terminal opens normally with no activation
- The user's `~/.zshrc` customizations (prompt, PATH, etc.) are still present

**Todo List**
1. Create `.vscode/.zshrc` with the following structure:
   - Source `~/.zshrc` if it exists (guard against double-sourcing with a sentinel variable)
   - Determine `PROJECT_DIR` from the script's own location (`${0:A:h}/..` resolves to workspace root from `.vscode/`)
   - `if [[ -d "$PROJECT_DIR/.venv" ]]; then` → `source "$PROJECT_DIR/.venv/bin/activate"`
   - `elif [[ -f "$PROJECT_DIR/.conda_config" ]]; then` → read the path, `conda activate <path>`

**Relevant Context**
- Current workspace has `.venv/` present (confirmed by subagent)
- `.conda_config` does not currently exist (future-proofing this branch)
- The `conda` command must be available in PATH for conda activation to work; if miniforge is used, it may require `conda init` output in `~/.zshrc`
- Double-sourcing `~/.zshrc` can cause problems — use a guard env var `ZSHRC_SOURCED`

**Status:** [x] done

---

## Relevant Files

| File | Action | Notes |
|------|--------|-------|
| `.vscode/settings.json` | Create | Workspace terminal profile; gitignored |
| `.vscode/.zshrc` | Create | Custom zsh startup; gitignored |
| `.gitignore` | No change | `.vscode/` already excluded |
| `.venv/pyvenv.cfg` | No change | Confirms venv path is `.venv/bin/activate` |

---

## Design Decisions

1. **`ZDOTDIR` approach vs `shellArgs`**: Using `ZDOTDIR` is cleaner than passing `-c` or script arguments to zsh. It participates in normal zsh startup sequencing and allows `~/.zshrc` to be sourced naturally.

2. **Sourcing `~/.zshrc` from project `.zshrc`**: The user's personal zsh config (prompt, PATH, conda init, etc.) must still load. We source it first with a guard to prevent recursion.

3. **`.conda_config` parsing**: The plan assumes the file contains the conda env path as plain text. If the format is different (YAML, INI, etc.), the extraction command in `.vscode/.zshrc` must be adjusted.

4. **Scope**: This is workspace-local only (`.vscode/` is gitignored). Other team members are unaffected.
