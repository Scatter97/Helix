# Helix

Current version: **v0.6.0**

Helix is an autonomous coding agent that can work on an existing repository or start in an empty project folder.

Helix v0.1.0 focuses on a transparent agent loop:

1. inspect the project,
2. create new files when needed,
3. modify existing files through Patch Forge v2,
4. run commands/builds/tests,
5. diagnose failures,
6. retry,
7. finish only after verification.

## Model routing

The default configuration defines four roles:

- `small`: lightweight local helper model
- `main`: normal local coding model
- `large`: local escalation model
- `remote`: optional OpenAI-compatible API backend

Remote failures fall back to configured local models.

## Patch Forge

Existing-file source mutations go through Patch Forge. Models issue a compact `patch_file` action; Helix calculates the current SHA-256 and builds the strict Patch Forge v2 document itself. Generated patches are saved under `.helix/patches/`, dry-run before application, checkpointed, and rejected if exact-match assumptions are wrong.

Configure the Patch Forge CLI path in `helix.toml`.

## Run

```powershell
python .\\helix.py --project "C:\\Path\\To\\Project" --task "Build a small notes app and verify it works."
```

For a remote OpenAI-compatible provider, fill in the remote model/base URL in `helix.toml`, enable it, and set the configured API-key environment variable.

## Safety in v0.1.0

- file paths are confined to the target project root
- `create_file` refuses to overwrite existing files
- existing-file modifications are delegated to Patch Forge
- Helix does not commit or push Git changes
- completion is rejected after code changes until verification succeeds

Helix is experimental software. Review diffs and run it only on repositories you are prepared to modify.


## Reliability in v0.2.1

Helix detects repeated no-progress actions and repeated unchanged failures. When a task stalls, it escalates from the main model to the configured large model and then to the optional remote model. If no configured model can make progress, Helix stops safely instead of looping forever.

`max_steps = 0` means there is no hard step limit. Stall detection remains active and becomes the normal protection against runaway loops.

Large new files can be created incrementally with `create_file_chunk` using `start`, `append`, and `finish` modes, avoiding a single oversized JSON action.

Patch Forge checkpoint directories are excluded from repository scans.

## Multiline tasks

At the interactive prompt, enter `:task`, paste or type as many lines as needed, then enter `:end` on its own line to submit.


## Creation reliability in v0.2.2

Helix normalizes absolute paths that still point inside the active project, making chunked creation tolerant of either `src/app.py` or an absolute in-project path. Paths outside the project remain rejected.

Large generated files should use approximately 2000-2500 characters per `create_file_chunk` action. The hard configured limit remains enforced, and oversized actions now report their actual size and explicit retry instructions.

Model requests are wrapped in an interruptible polling layer so Ctrl+C can return control to the interactive shell without waiting for the entire model HTTP timeout. The underlying abandoned request runs only in a daemon thread and cannot block process shutdown.

Multiline tasks are no longer printed in full a second time when execution starts; the task header displays only the first non-empty line.


## Unified Patch Forge writes in v0.3.0

Patch Forge now owns both creation of new project files and modification of existing project files.

For a new file, Helix uses `begin_file`, one or more `append_file` actions, and `finish_file`. The intermediate chunks are stored only under `.helix/staging`. When `finish_file` runs, Helix assembles the complete contents and sends them to Patch Forge using the v2 `action: create` format.

The final source file is therefore created by Patch Forge, not directly by Helix. Patch Forge provides create-only protection, dry-run validation, proposed diffs, checkpoints, and rollback behavior for those files.

Existing files continue to use SHA-256 protected `patch_file` edits through Patch Forge.

This also prevents models from needing to place an entire large source file inside a single `create_file` JSON action. Every new file, even a small one, uses the staged begin/append/finish protocol.


## Staged-work awareness in v0.3.1

Helix now treats unfinished staged files as first-class agent state. `begin_file` and `append_file` update the workspace revision and staged-file metadata so useful staging work is not mistaken for lack of progress.

The model payload includes each unfinished staged file, its current character count, and explicit guidance to continue with `append_file` or `finish_file` instead of rescanning the repository.

Calling `begin_file` for an already staged path is idempotent: Helix preserves the staged contents and tells the model to continue rather than treating the action as a failure.

`read_file` can also read the contents of `.helix/staging` when the final project file does not yet exist. After Patch Forge successfully creates the final file, the staged entry is removed from agent state.


## Model protocol self-repair in v0.3.2

Helix now diagnoses invalid model responses and asks the same model to repair its exact protocol error before escalating to another model. The repair request includes the parser or tool-schema diagnostic, the invalid response, and the original task-state payload including unfinished staged files.

The default repair budget is two attempts per model and is configurable with `runtime.model_repair_attempts`. Blank responses, malformed JSON, unknown tools, missing required arguments, unexpected arguments, and basic argument type errors can all produce targeted corrective feedback.

Protocol repair preserves staged-file context and explicitly instructs the model to continue its current work rather than restarting or rescanning the repository.

Helix also checks `append_file` payloads for substantial overlap with content already staged. Duplicate or overlapping chunks are rejected before they can be appended, reducing accidental duplicated source code before `finish_file` sends the completed file to Patch Forge.


## Failure-guided patch recovery in v0.3.3

Helix now preserves failed `patch_file` details as first-class task state. The next model request includes the failed path, Patch Forge stage, diagnostic output, and an explicit recovery instruction.

While a patch failure remains unresolved, `list_files` is blocked so the agent cannot fall back into repetitive repository scans. The agent is directed to read the failed file, inspect the exact error, and construct a corrected patch.

Insert operations also receive adjacent-source overlap protection. If an `insert_before` or `insert_after` payload repeats text that already exists immediately beside the anchor, Helix rejects the edit before Patch Forge runs and reports the duplicated region.

Python files modified through `patch_file` now receive a `python -m py_compile` verification command inside the Patch Forge workflow in addition to `git diff --check`. With rollback enabled, syntax-invalid Python edits can be restored automatically and the exact verification failure returned to Helix for another debugging iteration.


## Staged-work continuity in v0.3.4

Helix now locks autonomous creation onto the active staged file. While a file is unfinished, repository rescans, unrelated file reads, and attempts to begin another file are rejected with a targeted continuation diagnostic.

The model context explicitly includes `active_staged_file`, and the coding prompt instructs the model to continue that file using `append_file`, `read_file`, or `finish_file` instead of drifting into guessed or unrelated filenames.

Protocol guidance is also stricter. Invalid tool arguments must be corrected according to Helix's actual tool schema, and append chunks should remain comfortably below the hard payload limit. This specifically targets failures where a malformed response caused the model to switch from the current todo application into unrelated files such as `calculator.py`, `README.md`, or `pyproject.toml`.

A more radical raw-code/non-JSON file transport remains a possible future architecture change, but v0.3.4 first strengthens the existing protocol so its effect can be measured independently.


## Raw source transport in v0.3.5

v0.3.5 introduces `append_file_raw` for source-code creation. The model emits only a small JSON header containing the target path, followed by literal source text between `<<<HELIX_CONTENT` and `HELIX_CONTENT` markers. Quotes, triple-quoted strings, backslashes, f-strings, and multiline source no longer need to be escaped as JSON.

The existing `append_file` tool remains available for compatibility, while `append_file_raw` is the preferred tool for source code and other quote-heavy multiline content. Existing staged-file size limits and duplicate-overlap checks also apply to raw chunks.

## Explicit LARGE-model handoff

When LARGE is invoked, Helix now constructs an explicit escalation handoff from shared agent state. It summarizes the original task, changed files, staged work, the active staged file, the most recently completed file, recent actions, tool failures, verification status, previous model errors, and a recommended continuation action.

The complete current Helix state is also included after the summary. LARGE therefore continues from MAIN's observable work and tool results rather than restarting the task. Hidden model reasoning is not transferred or required.

## Post-finish continuity

Helix records `last_completed_file` after a successful Patch Forge creation and includes a post-finish continuation instruction in subsequent model context. The model is directed toward remaining requirements, testing, or verification instead of immediately rereading or rescanning work that was just completed.


## Two-phase raw generation in v0.3.6

Ollama normally remains in structured JSON mode for Helix action selection. When the selected action is `append_file_raw`, the first response contains only the target path. Helix then performs a second model call with Ollama JSON formatting disabled and requests only the next literal source-code chunk.

This preserves reliable structured tool selection while completely separating source code from JSON escaping. Triple-quoted strings, ordinary quotes, backslashes, f-strings, and multiline source are therefore no longer embedded inside a JSON action.

The two-phase design also keeps the existing MAIN-to-LARGE escalation handoff. Whichever model owns the current step receives the shared Helix task state, and raw source generation receives focused staged-file context for the exact target file.


## Post-finish planning in v0.3.7

Helix now treats successful `finish_file` as a planning transition rather than returning immediately to unrestricted repository exploration. `post_finish_pending` records that a file was just completed and that the next action should advance the original task.

While this state is active, redundant `list_files` calls and immediate rereads of `last_completed_file` are rejected. Beginning a new required file, patching code, or running an implementation/verification command clears the transition state.

Helix also derives a lightweight `remaining_requirements_instruction` from observable task state. For example, if the original task explicitly requested automated tests but no test-like file has been completed, the model is told that tests remain. Missing final verification is also surfaced.

The MAIN-to-LARGE escalation handoff now carries `post_finish_pending` and the remaining-requirements guidance so LARGE can continue from the next unfinished requirement instead of restarting completed work or repeating repository scans.


## v0.6.0 model architecture

Helix uses three explicit local roles:

1. **Architect** ? produces the exhaustive pre-coding development blueprint
   and is reloaded as a recovery consultant after repeated stalls.
2. **Orchestrator** ? chooses Helix actions and follows the architect plan.
3. **Coder** ? writes and repairs literal source code only.

The old LARGE fallback is no longer part of the normal runtime architecture.

Before architect work, Helix unloads the orchestrator and coder. After the
architect produces its plan or recovery guide, the architect is unloaded and
normal orchestration resumes.

Architect artifacts are stored under `.helix/`.


## v0.6.1 architect requirement provenance

The architect now separates project information into four provenance classes:

- EXPLICIT ? directly requested by the user and binding.
- DERIVED ? logically required to make an explicit requirement work.
- DESIGN ? non-binding implementation recommendation.
- OPTIONAL ? optional enhancement.

Only EXPLICIT and genuinely necessary DERIVED requirements may enter the
execution contract.

This prevents the architect from silently converting choices such as package
names, storage paths, optional flags, module structures, or output wording into
user requirements.


## v0.6.2 architect transport reliability

Architect generation now has dedicated transport diagnostics and progress
reporting.

Changes include:

- architect heartbeat output every 10 seconds,
- one automatic retry when Ollama returns empty final content,
- Ollama content/thinking character-count metadata,
- generated-token and done-reason diagnostics when available,
- development-plan size reporting,
- architect execution-contract size reporting.

Helix never substitutes Ollama thinking content for the architect's final
development blueprint. Thinking metadata is used only to diagnose transport
behavior.


## v0.6.3 long architect generation and model diagnostics

The architect now supports long multi-part blueprints.

When Ollama returns `done_reason=length`, Helix does not accept the truncated
plan as complete. It requests a continuation and assembles the final blueprint
before saving it.

Model output budgets are now configurable through `num_predict`.

Recommended defaults:

- architect: 12000
- orchestrator: 2048
- coder: 8192

All Ollama model roles now report transport metadata when they return empty
responses, including final-content size, thinking size, generation count,
done reason, and whether JSON mode was active.


## v0.6.4 role-specific context windows

Helix now configures Ollama context-window size independently for each
model role.

Recommended local defaults:

- architect: num_ctx=32768, num_predict=12000
- orchestrator: num_ctx=8192, num_predict=2048
- coder: num_ctx=16384, num_predict=8192

Architect continuation calls retain only the last 3000 characters of the
previous part instead of 8000 characters, reducing context waste.

Architect diagnostics now report prompt tokens, generated tokens, configured
context size, and Ollama's completion reason.
