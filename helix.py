from __future__ import annotations

import argparse
import contextlib
import csv
import io
import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import sys
import time
import tempfile
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VERSION = "0.6.4"


class HelixError(RuntimeError):
    pass


class RawGenerationError(HelixError):
    """Raised after raw source generation exhausts its own retry budget."""



@dataclass
class ModelSpec:
    provider: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
    timeout_seconds: int = 180
    enabled: bool = True
    raw_target_chars: int = 7000
    raw_max_chars: int = 12000
    num_predict: int = 4096
    num_ctx: int = 8192


@dataclass
class HelixConfig:
    patchforge_cli: str
    max_steps: int
    command_timeout_seconds: int
    models: dict[str, ModelSpec]
    stall_escalation_steps: int = 3
    stall_stop_steps: int = 8
    max_create_chunk_chars: int = 3500
    model_repair_attempts: int = 2
    raw_generation_attempts: int = 2
    model_circuit_breaker_failures: int = 3
    architect_recovery_stall_steps: int = 3
    architect_max_recoveries: int = 3

    @classmethod
    def load(cls, path: Path) -> "HelixConfig":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        runtime = data.get("runtime", {})
        model_data = data.get("models", {})
        models: dict[str, ModelSpec] = {}
        for role, raw in model_data.items():
            models[role] = ModelSpec(
                provider=str(raw.get("provider", "ollama")),
                model=str(raw.get("model", "")),
                base_url=str(raw.get("base_url", "")),
                api_key_env=str(raw.get("api_key_env", "")),
                timeout_seconds=int(raw.get("timeout_seconds", 180)),
                enabled=bool(raw.get("enabled", True)),
                raw_target_chars=max(
                    1500,
                    int(raw.get("raw_target_chars", 7000)),
                ),
                raw_max_chars=max(
                    3500,
                    int(raw.get("raw_max_chars", 12000)),
                ),
                num_predict=max(
                    256,
                    int(raw.get("num_predict", 4096)),
                ),
                num_ctx=max(
                    2048,
                    int(raw.get("num_ctx", 8192)),
                ),
            )
        return cls(
            patchforge_cli=str(runtime.get("patchforge_cli", "")),
            max_steps=int(runtime.get("max_steps", 40)),
            command_timeout_seconds=int(runtime.get("command_timeout_seconds", 180)),
            models=models,
        stall_escalation_steps=max(
            1,
            int(runtime.get("stall_escalation_steps", 3)),
        ),
        stall_stop_steps=max(
            2,
            int(runtime.get("stall_stop_steps", 8)),
        ),
        max_create_chunk_chars=max(
            500,
            int(runtime.get("max_create_chunk_chars", 3500)),
        ),
        model_repair_attempts=max(
            0,
            int(runtime.get("model_repair_attempts", 2)),
        ),
        raw_generation_attempts=max(
            0,
            int(runtime.get("raw_generation_attempts", 2)),
        ),
        model_circuit_breaker_failures=max(
            1,
            int(runtime.get("model_circuit_breaker_failures", 3)),
        ),
        architect_recovery_stall_steps=max(
            1,
            int(runtime.get("architect_recovery_stall_steps", 3)),
        ),
        architect_max_recoveries=max(
            0,
            int(runtime.get("architect_max_recoveries", 3)),
        ),
        )


@dataclass
class StepRecord:
    step: int
    model_role: str
    tool: str
    ok: bool
    summary: str


@dataclass
class AgentState:
    task: str
    project_root: Path
    step: int = 0
    coding_failures: int = 0
    model_failures: int = 0
    verification_passed: bool = False
    last_verification_command: str = ""
    changed_files: set[str] = field(default_factory=set)
    history: list[StepRecord] = field(default_factory=list)
    stall_steps: int = 0
    stall_level: int = 0
    workspace_revision: int = 0
    seen_actions: set[str] = field(default_factory=set)
    last_failure_signature: str = ""
    staged_files: dict[str, int] = field(default_factory=dict)
    last_tool_failure: dict[str, Any] | None = None
    last_completed_file: str = ""
    post_finish_pending: bool = False
    development_plan: str = ""
    architect_contract: str = ""
    architect_recovery_guidance: str = ""
    architect_recoveries: int = 0
    last_architect_recovery_step: int = -1000

    def add(self, role: str, tool: str, ok: bool, summary: str) -> None:
        self.history.append(
            StepRecord(
                step=self.step,
                model_role=role,
                tool=tool,
                ok=ok,
                summary=summary[:4000],
            )
        )

    def recent_context(self, limit: int = 12) -> list[dict[str, Any]]:
        return [
            {
                "step": item.step,
                "model_role": item.model_role,
                "tool": item.tool,
                "ok": item.ok,
                "summary": item.summary,
            }
            for item in self.history[-limit:]
        ]


@dataclass
class AgentPermissions:
    write: bool = True
    execute: bool = True


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def normalize_relative(self, value: str) -> str:
        path = self.resolve(value)
        return path.relative_to(self.root).as_posix()

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise HelixError(f"Path escapes project root: {relative}") from exc
        return candidate

    def list_files(self, max_files: int = 300) -> list[str]:
        ignored = {
            ".git",
            ".helix",
            ".patchforge",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".nox",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "dist",
            "build",
        }
        result: list[str] = []
        for path in self.root.rglob("*"):
            if any(part in ignored for part in path.parts):
                continue
            if path.is_file():
                result.append(path.relative_to(self.root).as_posix())
                if len(result) >= max_files:
                    break
        return sorted(result)

    def read_file(self, relative: str, start_line: int = 1, end_line: int | None = None) -> str:
        path = self.resolve(relative)

        if not path.is_file():
            _, stage, normalized = self._staging_path(relative)
            if stage.is_file():
                path = stage
            else:
                raise HelixError(f"File does not exist: {normalized}")

        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        start = max(1, start_line)
        end = len(lines) if end_line is None else min(len(lines), end_line)
        if end < start:
            return ""
        return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))

    def search_text(self, query: str, max_results: int = 50) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for relative in self.list_files(max_files=1000):
            path = self.resolve(relative)
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(lines, 1):
                if query.lower() in line.lower():
                    results.append({"path": relative, "line": number, "text": line[:500]})
                    if len(results) >= max_results:
                        return results
        return results

    def _staging_path(self, relative: str) -> tuple[Path, Path, str]:
        target = self.resolve(relative)
        normalized = target.relative_to(self.root).as_posix()

        staging_root = (self.root / ".helix" / "staging").resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
        stage = (staging_root / f"{normalized}.part").resolve()

        try:
            stage.relative_to(staging_root)
        except ValueError as exc:
            raise HelixError(f"Invalid staged file path: {normalized}") from exc

        return target, stage, normalized

    def create_file_chunk(
        self,
        relative: str,
        content: str,
        mode: str,
    ) -> str | None:
        target, stage, normalized = self._staging_path(relative)

        if target.exists():
            raise HelixError(
                f"Refusing to create existing file {normalized}; use patch_file for modifications."
            )

        mode = mode.lower().strip()

        if mode == "start":
            if stage.exists():
                raise HelixError(
                    f"A staged file already exists for {normalized}; continue with append_file or finish_file."
                )
            stage.parent.mkdir(parents=True, exist_ok=True)
            stage.write_text(content, encoding="utf-8", newline="\n")
            return None

        if mode not in {"append", "finish"}:
            raise HelixError(
                "Staged file mode must be start, append, or finish."
            )

        if not stage.is_file():
            raise HelixError(
                f"No staged file exists for {normalized}; call begin_file first."
            )

        if content:
            with stage.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(content)

        if mode == "append":
            return None

        return stage.read_text(encoding="utf-8")

    def replace_staged_file(
        self,
        relative: str,
        content: str,
    ) -> None:
        target, stage, normalized = self._staging_path(relative)

        if target.exists():
            raise HelixError(
                f"Refusing to replace staged content for existing file "
                f"{normalized}."
            )

        if not stage.is_file():
            raise HelixError(
                f"No staged file exists for {normalized}; "
                "call begin_file first."
            )

        stage.write_text(
            content,
            encoding="utf-8",
            newline="\n",
        )

    def staged_file_characters(self, relative: str) -> int | None:
        _, stage, _ = self._staging_path(relative)
        if not stage.is_file():
            return None
        return len(stage.read_text(encoding="utf-8"))

    def staged_append_overlap(
        self,
        relative: str,
        content: str,
        min_overlap: int = 80,
    ) -> int:
        _, stage, _ = self._staging_path(relative)
        if not stage.is_file() or not content:
            return 0

        existing = stage.read_text(encoding="utf-8")
        if not existing:
            return 0

        if len(content) >= min_overlap and content in existing:
            return len(content)

        largest = min(len(existing), len(content))
        for size in range(largest, min_overlap - 1, -1):
            if existing.endswith(content[:size]):
                return size

        return 0

    def clear_staged_file(self, relative: str) -> None:
        _, stage, _ = self._staging_path(relative)
        if stage.exists():
            stage.unlink()

    def hash_file(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.is_file():
            raise HelixError(f"File does not exist: {relative}")
        return hashlib.sha256(path.read_bytes()).hexdigest()


class CommandRunner:
    def __init__(self, root: Path, default_timeout: int):
        self.root = root
        self.default_timeout = default_timeout

    def run(self, command: str, timeout_seconds: int | None = None) -> dict[str, Any]:
        blocked = (
            "git push",
            "git commit",
            "git reset --hard",
            "git clean -",
            "shutdown",
            "format ",
            "rm -rf",
            "remove-item -recurse",
            "del /s",
        )
        lowered = command.lower()
        if any(token in lowered for token in blocked):
            return {
                "ok": False,
                "exit_code": None,
                "stdout": "",
                "stderr": "Command blocked by Helix safety policy.",
                "duration_seconds": 0,
                "timed_out": False,
            }
        timeout = timeout_seconds or self.default_timeout
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                cwd=self.root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-12000:],
                "stderr": proc.stderr[-12000:],
                "duration_seconds": round(time.monotonic() - started, 2),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "exit_code": None,
                "stdout": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
                "duration_seconds": round(time.monotonic() - started, 2),
                "timed_out": True,
            }


class PatchForgeAdapter:
    def __init__(self, project_root: Path, patchforge_cli: str):
        self.project_root = project_root
        self.patchforge_cli = Path(patchforge_cli).resolve() if patchforge_cli else None
        self.patch_dir = project_root / ".helix" / "patches"
        self.patch_dir.mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        return bool(self.patchforge_cli and self.patchforge_cli.exists())

    @staticmethod
    def infrastructure_failure(result: dict[str, Any]) -> bool:
        combined = "\n".join(
            str(result.get(key, ""))
            for key in (
                "stdout",
                "stderr",
                "check_output",
                "error",
            )
        ).lower()

        markers = (
            "unicodeencodeerror",
            "charmap",
            "cp1252",
            "codec can't encode",
            "codec cannot encode",
        )
        return any(marker in combined for marker in markers)


    def create_file(
        self,
        relative_path: str,
        content: str,
        attempt: int,
    ) -> dict[str, Any]:
        root = self.project_root.resolve()
        target = (root / relative_path).resolve()

        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HelixError(
                f"Path escapes project root: {relative_path}"
            ) from exc

        normalized = target.relative_to(root).as_posix()

        if target.exists():
            raise HelixError(
                f"Cannot create existing file {normalized}; use patch_file for modifications."
            )

        patch = {
            "version": 2,
            "name": f"Helix create {normalized}",
            "files": [
                {
                    "path": normalized,
                    "action": "create",
                    "content": content,
                }
            ],
            "options": {
                "rollback_on_verify_failure": True
            },
        }

        return self.apply(patch, attempt)

    @staticmethod
    def _adjacent_insert_overlap(
        source_text: str,
        anchor: str,
        content: str,
        op: str,
        min_overlap: int = 20,
    ) -> str:
        if not anchor or not content:
            return ""

        if source_text.count(anchor) != 1:
            return ""

        anchor_pos = source_text.find(anchor)

        if op == "insert_after":
            adjacent = source_text[anchor_pos + len(anchor):]
            largest = min(len(adjacent), len(content))
            for size in range(largest, min_overlap - 1, -1):
                if content[:size] == adjacent[:size]:
                    return content[:size]

        if op == "insert_before":
            adjacent = source_text[:anchor_pos]
            largest = min(len(adjacent), len(content))
            for size in range(largest, min_overlap - 1, -1):
                if content[-size:] == adjacent[-size:]:
                    return content[-size:]

        return ""

    def apply_edit(
        self,
        relative_path: str,
        op: str,
        attempt: int,
        *,
        old: str | None = None,
        new: str | None = None,
        anchor: str | None = None,
        content: str | None = None,
        expected_matches: int = 1,
    ) -> dict[str, Any]:
        source = (self.project_root / relative_path).resolve()
        try:
            source.relative_to(self.project_root.resolve())
        except ValueError as exc:
            raise HelixError(f"Path escapes project root: {relative_path}") from exc

        if not source.is_file():
            raise HelixError(
                f"Cannot patch missing file {relative_path}; use create_file first."
            )

        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source_text = source.read_text(encoding="utf-8")


        if op == "replace":
            if not isinstance(old, str) or not isinstance(new, str):
                raise HelixError("replace requires string old and new arguments.")
            edit = {
                "op": "replace",
                "old": old,
                "new": new,
                "expected_matches": expected_matches,
            }
        elif op in {"insert_before", "insert_after"}:
            if not isinstance(anchor, str) or not isinstance(content, str):
                raise HelixError(f"{op} requires string anchor and content arguments.")

            overlap = self._adjacent_insert_overlap(
                source_text,
                anchor,
                content,
                op,
            )
            if overlap:
                return {
                    "ok": False,
                    "stage": "precheck",
                    "path": relative_path,
                    "error": (
                        "Patch content duplicates source text immediately "
                        f"adjacent to the {op} anchor. Remove the duplicated "
                        "existing text and retry."
                    ),
                    "duplicate_text": overlap[:2000],
                }

            edit = {
                "op": op,
                "anchor": anchor,
                "content": content,
                "expected_matches": expected_matches,
            }
        else:
            raise HelixError(f"Unsupported Patch Forge edit operation: {op}")

        verify_commands = [
            {
                "type": "command",
                "name": "Check resulting diff",
                "command": "git diff --check",
                "expect_exit": 0,
            }
        ]

        if source.suffix.lower() == ".py":
            verify_commands.append(
                {
                    "type": "command",
                    "name": f"Compile {relative_path}",
                    "command": (
                        "python -m py_compile "
                        + json.dumps(relative_path)
                    ),
                    "expect_exit": 0,
                }
            )

        patch = {
            "version": 2,
            "name": f"Helix attempt {attempt:03d} - {relative_path}",
            "preflight": [
                {
                    "type": "sha256",
                    "name": f"Check {relative_path} baseline",
                    "path": relative_path,
                    "equals": digest,
                }
            ],
            "files": [
                {
                    "path": relative_path,
                    "sha256": digest,
                    "edits": [edit],
                }
            ],
            "verify": verify_commands,
            "options": {
                "rollback_on_verify_failure": True
            }
        }

        return self.apply(patch, attempt)

    def apply(self, patch: dict[str, Any], attempt: int) -> dict[str, Any]:
        if not self.available():
            raise HelixError(
                "Patch Forge CLI not configured or not found. Set runtime.patchforge_cli in helix.toml."
            )
        patch_path = self.patch_dir / f"attempt-{attempt:03d}.json"
        patch_path.write_text(json.dumps(patch, indent=2), encoding="utf-8")

        base = [
            sys.executable,
            str(self.patchforge_cli),
            "--repo",
            str(self.project_root),
        ]
        subprocess_env = os.environ.copy()
        subprocess_env["PYTHONUTF8"] = "1"
        subprocess_env["PYTHONIOENCODING"] = "utf-8"

        check = subprocess.run(
            base + ["check", str(patch_path)],
            cwd=self.patchforge_cli.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=subprocess_env,
        )
        if check.returncode != 0:
            return {
                "ok": False,
                "stage": "check",
                "patch_path": str(patch_path),
                "stdout": check.stdout[-12000:],
                "stderr": check.stderr[-12000:],
            }

        apply = subprocess.run(
            base + ["apply", str(patch_path)],
            cwd=self.patchforge_cli.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=subprocess_env,
        )
        return {
            "ok": apply.returncode == 0,
            "stage": "apply",
            "patch_path": str(patch_path),
            "check_output": check.stdout[-12000:],
            "stdout": apply.stdout[-12000:],
            "stderr": apply.stderr[-12000:],
        }


_INTERRUPTIBLE_WAIT_HOOK = None


def interruptible_call(fn):
    results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            results.put((True, fn()))
        except BaseException as exc:
            results.put((False, exc))

    thread = threading.Thread(
        target=worker,
        name="helix-model-request",
        daemon=True,
    )
    thread.start()

    wait_started = time.monotonic()
    last_heartbeat = wait_started

    while True:
        try:
            ok, value = results.get(timeout=0.1)
            break
        except queue.Empty:
            hook = _INTERRUPTIBLE_WAIT_HOOK
            now = time.monotonic()

            if (
                hook is not None
                and now - last_heartbeat >= 1.0
            ):
                try:
                    hook(now - wait_started)
                except Exception:
                    pass
                last_heartbeat = now

            continue

    if ok:
        return value
    raise value


class ModelBackend:
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
    ) -> str:
        raise NotImplementedError

    def unload(self) -> None:
        """Release model resources when supported by the provider."""
        return None


class OllamaBackend(ModelBackend):
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
    ) -> str:
        base = (self.spec.base_url or "http://localhost:11434").rstrip("/")
        payload = {
            "model": self.spec.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "num_predict": self.spec.num_predict,
                "num_ctx": self.spec.num_ctx,
            },
        }

        if json_mode:
            payload["format"] = "json"
        request = urllib.request.Request(
            base + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.spec.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(
                "utf-8",
                errors="replace",
            )
            raise HelixError(
                f"Ollama HTTP {exc.code}: {body[:1500]}"
            ) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            raise HelixError(
                f"Ollama request failed: {exc}"
            ) from exc
        message = data.get("message", {})

        content = str(
            message.get("content", "") or ""
        )

        thinking = str(
            message.get("thinking", "") or ""
        )

        # Preserve transport diagnostics without exposing or treating
        # model thinking as the final answer.
        self.last_response_metadata = {
            "content_chars": len(content),
            "thinking_chars": len(thinking),
            "done": data.get("done"),
            "done_reason": data.get("done_reason"),
            "prompt_eval_count": data.get(
                "prompt_eval_count"
            ),
            "eval_count": data.get("eval_count"),
            "total_duration": data.get(
                "total_duration"
            ),
            "load_duration": data.get(
                "load_duration"
            ),
        }

        return content

    def unload(self) -> None:
        base = (
            self.spec.base_url
            or "http://localhost:11434"
        ).rstrip("/")

        payload = {
            "model": self.spec.model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }

        request = urllib.request.Request(
            base + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=min(self.spec.timeout_seconds, 30),
            ) as response:
                response.read()
        except Exception:
            # Unloading is resource management, not task correctness.
            # A failed unload must never destroy an otherwise valid task.
            pass


class OpenAICompatibleBackend(ModelBackend):
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
    ) -> str:
        if not self.spec.base_url:
            raise HelixError("Remote model base_url is empty.")
        api_key = os.environ.get(self.spec.api_key_env, "") if self.spec.api_key_env else ""
        if self.spec.api_key_env and not api_key:
            raise HelixError(f"Missing environment variable {self.spec.api_key_env}.")
        base = self.spec.base_url.rstrip("/")
        url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        payload = {
            "model": self.spec.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.spec.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise HelixError(f"Remote model HTTP {exc.code}: {body[:1000]}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise HelixError(f"Remote model request failed: {exc}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise HelixError("Remote model returned no choices.")
        return str(choices[0].get("message", {}).get("content", ""))


def build_raw_content_prompt(
    *,
    original_user: str,
    path: str,
    target_chars: int = 7000,
    max_chars: int = 12000,
) -> str:
    try:
        state = json.loads(original_user)
    except json.JSONDecodeError:
        state = {}

    if not isinstance(state, dict):
        state = {}

    task = state.get("task", "")
    staged_files = state.get("staged_files", [])
    active_staged_file = state.get("active_staged_file")
    recent_history = state.get("recent_history", [])
    last_completed_file = state.get("last_completed_file", "")
    changed_files = state.get("changed_files", [])
    remaining_requirements = state.get(
        "remaining_requirements_instruction",
        "",
    )

    filename = Path(path).name.lower()
    is_test_file = (
        filename.startswith("test_")
        or filename.endswith("_test.py")
    )

    target_instruction = ""

    if is_test_file:
        target_instruction = (
            "This target is a pytest test file. Generate real "
            "pytest-discoverable test_* functions. Import and test the "
            "completed application implementation rather than rewriting "
            "the application in this file."
        )

        if last_completed_file:
            target_instruction += (
                f" The implementation to test is "
                f"{last_completed_file}."
            )

    context = {
        "task": task,
        "target_path": path,
        "active_staged_file": active_staged_file,
        "staged_files": staged_files,
        "last_completed_file": last_completed_file,
        "changed_files": changed_files,
        "remaining_requirements": remaining_requirements,
        "target_instruction": target_instruction,
        "architect_contract": state.get(
            "architect_contract",
            "",
        ),
        "architect_recovery_guidance": state.get(
            "architect_recovery_guidance",
            "",
        ),
        "recent_history": (
            recent_history[-6:]
            if isinstance(recent_history, list)
            else []
        ),
    }

    return (
        "HELIX RAW FILE CONTENT GENERATION\n\n"
        f"Generate the NEXT literal source-code chunk for: {path}\n\n"
        "Return ONLY literal file content.\n"
        "Do NOT return JSON.\n"
        "Do NOT return markdown fences.\n"
        "Do NOT include <<<HELIX_CONTENT or HELIX_CONTENT markers.\n"
        "Do NOT explain the code.\n"
        "Do NOT repeat source already staged.\n"
        f"Preferred response size: up to about {target_chars} characters.\n"
        f"Absolute maximum response size: {max_chars} characters.\n"
        "If the complete remaining file fits comfortably within the preferred "
        "response size, return the complete remaining file in this response.\n"
        "If it does not fit, return one coherent continuation chunk and stop "
        "at a clean source-code boundary.\n"
        "Continue exactly from the current staged file state.\n\n"
        "CURRENT TASK CONTEXT:\n"
        + json.dumps(context, indent=2, ensure_ascii=False)
    )


class ModelRouter:
    def __init__(self, config: HelixConfig):
        self.config = config
        self.last_model_failures = 0
        self.event_sink = None
        self.task_model_failures: dict[str, int] = {}
        self.task_unhealthy_models: set[str] = set()

    def reset_task_health(self) -> None:
        self.task_model_failures.clear()
        self.task_unhealthy_models.clear()

    def _record_model_failure(self, role: str) -> None:
        count = self.task_model_failures.get(role, 0) + 1
        self.task_model_failures[role] = count

        if count >= self.config.model_circuit_breaker_failures:
            if role not in self.task_unhealthy_models:
                self.task_unhealthy_models.add(role)
                self._model_log(
                    role,
                    (
                        "disabled for the current task after "
                        f"{count} consecutive unusable response(s)"
                    ),
                    kind="model_circuit_breaker",
                )

    def _record_model_success(self, role: str) -> None:
        self.task_model_failures[role] = 0

    def _event(self, kind: str, **data: Any) -> bool:
        if self.event_sink is None:
            return False
        try:
            return bool(self.event_sink(kind, data))
        except Exception:
            return False

    def _model_log(self, role: str, message: str, *, kind: str = "model") -> None:
        handled = self._event(kind, role=role, message=message)
        if not handled:
            print(f"[MODEL] {role} {message}")

    def choose_role(self, state: AgentState) -> str:
        if self._new_role_architecture():
            return "orchestrator"

        # Legacy configuration compatibility.
        if state.stall_level >= 2 and self._enabled("remote"):
            return "remote"
        if state.stall_level >= 1:
            if self._enabled("large"):
                return "large"
            if self._enabled("remote"):
                return "remote"

        failures = state.coding_failures
        if failures >= 5 and self._enabled("remote"):
            return "remote"
        if failures >= 3 and self._enabled("large"):
            return "large"
        return "main"

    def _enabled(self, role: str) -> bool:
        if role in self.task_unhealthy_models:
            return False
        spec = self.config.models.get(role)
        return bool(spec and spec.enabled and spec.model)

    def _new_role_architecture(self) -> bool:
        return (
            "architect" in self.config.models
            and "orchestrator" in self.config.models
            and "coder" in self.config.models
        )

    def unload_role(self, role: str) -> None:
        spec = self.config.models.get(role)

        if not spec or not spec.enabled or not spec.model:
            return

        try:
            self.backend(role).unload()
            self._model_log(
                role,
                "unloaded",
                kind="model_unload",
            )
        except Exception:
            pass

    def unload_roles(self, roles) -> None:
        for role in roles:
            self.unload_role(role)

    def backend(self, role: str) -> ModelBackend:
        spec = self.config.models.get(role)
        if not spec or not spec.enabled or not spec.model:
            raise HelixError(f"Model role '{role}' is not configured.")
        provider = spec.provider.lower()
        if provider == "ollama":
            return OllamaBackend(spec)
        if provider == "openai_compatible":
            return OpenAICompatibleBackend(spec)
        raise HelixError(f"Unsupported provider: {spec.provider}")

    @staticmethod
    def _unwrap_single_markdown_fence(content: str) -> tuple[str, bool]:
        stripped = content.strip()
        if not stripped.startswith("```"):
            return content, False
        lines = stripped.splitlines()
        if len(lines) < 3:
            raise HelixError("Raw source generation returned an incomplete markdown fence.")
        first = lines[0].strip()
        last = lines[-1].strip()
        if not re.fullmatch(r"```[A-Za-z0-9_+.#-]*", first):
            raise HelixError("Raw source generation returned an unsupported markdown fence.")
        if last != "```":
            raise HelixError("Raw source generation returned a markdown fence without a matching closing fence.")
        body = "\n".join(lines[1:-1])
        if "```" in body:
            raise HelixError("Raw source generation returned nested or multiple markdown code fences.")
        return body, True

    def _validate_raw_content(
        self,
        raw: str,
        max_chars: int | None = None,
    ) -> tuple[str, bool]:
        content, unwrapped = self._unwrap_single_markdown_fence(raw)
        if not content.strip():
            raise HelixError("Raw source generation returned an empty content body.")
        if "<<<HELIX_CONTENT" in content or "\nHELIX_CONTENT" in content:
            raise HelixError("Raw source generation must return only literal file content without HELIX_CONTENT markers.")
        if not content.endswith("\n"):
            content += "\n"
        effective_max = (
            int(max_chars)
            if max_chars is not None
            else self.config.max_create_chunk_chars
        )

        if len(content) > effective_max:
            raise HelixError(
                f"Raw generated chunk is too large: {len(content)} characters. "
                f"Hard maximum is {effective_max}. "
                "Generate a smaller coherent source response."
            )
        return content, unwrapped

    def _complete_raw_content(
        self,
        backend: ModelBackend,
        *,
        role: str = "main",
        system: str,
        original_user: str,
        path: str,
    ) -> str:
        spec = self.config.models.get(role)
        hard_max = (
            spec.raw_max_chars
            if spec is not None
            else 12000
        )
        target_chars = min(
            spec.raw_target_chars if spec is not None else 7000,
            hard_max,
        )

        max_retries = self.config.raw_generation_attempts
        retry_error = ""
        retry_raw = ""
        self._event("raw_start", role=role, path=path, max_retries=max_retries)
        for attempt_index in range(max_retries + 1):
            prompt = build_raw_content_prompt(
                original_user=original_user,
                path=path,
                target_chars=target_chars,
                max_chars=hard_max,
            )

            if attempt_index:
                prompt += (
                    "\n\nRAW GENERATION RETRY\n"
                    f"Previous raw-body error: {retry_error}\n"
                    "The append_file_raw action header has ALREADY been accepted. "
                    "Do not return JSON and do not choose another tool. "
                    f"Return only the corrected literal source continuation for {path}. "
                    "Do not use markdown fences. Do not repeat source that is already staged."
                )
                if retry_raw:
                    prompt += "\nThe previous invalid raw response began with:\n" + retry_raw[:500]
            try:
                raw = interruptible_call(
                    lambda: backend.complete(
                        system,
                        prompt,
                        json_mode=False,
                    )
                )

                if not raw.strip():
                    metadata = getattr(
                        backend,
                        "last_response_metadata",
                        {},
                    )

                    self._model_log(
                        role,
                        (
                            "empty raw response; "
                            f"content_chars={metadata.get('content_chars', 0)}, "
                            f"thinking_chars={metadata.get('thinking_chars', 0)}, "
                            f"prompt_eval_count={metadata.get('prompt_eval_count')}, "
                            f"eval_count={metadata.get('eval_count')}, "
                            f"done_reason={metadata.get('done_reason')!r}, "
                            "json_mode=False"
                        ),
                        kind="model_empty_response",
                    )
            except Exception as exc:
                retry_error = f"Raw model request failed: {exc}"
            else:
                try:
                    content, unwrapped = self._validate_raw_content(
                        raw,
                        max_chars=hard_max,
                    )
                    if unwrapped:
                        self._model_log(
                            role,
                            f"raw source for {path}: automatically removed one outer markdown code fence",
                            kind="raw_normalized",
                        )
                    self._event(
                        "raw_success",
                        role=role,
                        path=path,
                        characters=len(content),
                        attempt=attempt_index,
                    )
                    return content
                except Exception as exc:
                    retry_error = str(exc)
                    retry_raw = raw
            if attempt_index < max_retries:
                retry_number = attempt_index + 1

                old_target = target_chars
                target_chars = max(
                    1800,
                    int(target_chars * 0.65),
                )

                self._model_log(
                    role,
                    (
                        f"raw-generation retry {retry_number}/{max_retries} "
                        f"for {path}: {retry_error} "
                        f"[target {old_target} -> {target_chars} chars]"
                    ),
                    kind="raw_retry",
                )
        raise RawGenerationError(
            f"Raw generation for {path} failed after {max_retries} retry attempt(s): {retry_error}"
        )

    def action_with_fallback(
        self,
        role: str,
        system: str,
        user: str,
        coder_system: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        order = [role]

        if self._new_role_architecture():
            # The coder never participates in action/tool selection.
            # Architect is advisory only and is invoked explicitly by Core.
            for fallback in ("orchestrator", "remote"):
                if (
                    fallback not in order
                    and self._enabled(fallback)
                ):
                    order.append(fallback)
        else:
            # Legacy compatibility.
            for fallback in ("main", "large", "remote"):
                if (
                    fallback not in order
                    and self._enabled(fallback)
                ):
                    order.append(fallback)

        self.last_model_failures = 0
        errors: list[str] = []

        for candidate in order:
            backend = self.backend(candidate)
            request_user = user

            if candidate == "large":
                request_user = build_escalation_handoff(
                    original_user=user,
                    previous_errors=errors,
                )

            final_error = ""

            for repair_index in range(self.config.model_repair_attempts + 1):
                try:
                    raw = interruptible_call(
                        lambda: backend.complete(
                            system,
                            request_user,
                            json_mode=True,
                        )
                    )
                except Exception as exc:
                    self.last_model_failures += 1
                    final_error = f"Model request failed: {exc}"
                    self._record_model_failure(candidate)
                    self._model_log(candidate, f"request failed: {exc}", kind="model_failure")
                    break

                try:
                    if not raw.strip():
                        metadata = getattr(
                            backend,
                            "last_response_metadata",
                            {},
                        )

                        diagnostic = (
                            "empty response; "
                            f"content_chars={metadata.get('content_chars', 0)}, "
                            f"thinking_chars={metadata.get('thinking_chars', 0)}, "
                            f"prompt_eval_count={metadata.get('prompt_eval_count')}, "
                            f"eval_count={metadata.get('eval_count')}, "
                            f"done_reason={metadata.get('done_reason')!r}, "
                            "json_mode=True"
                        )

                        self._model_log(
                            candidate,
                            diagnostic,
                            kind="model_empty_response",
                        )

                    action = extract_action(raw)
                    validate_action(action)

                    if action.get("tool") == "append_file_raw":
                        args = action.get("args", {})
                        path = str(args.get("path", ""))

                        raw_role = (
                            "coder"
                            if self._new_role_architecture()
                            else candidate
                        )

                        raw_backend = (
                            self.backend(raw_role)
                            if raw_role != candidate
                            else backend
                        )

                        try:
                            raw_content = self._complete_raw_content(
                                raw_backend,
                                role=raw_role,
                                system=(
                                    coder_system
                                    if (
                                        raw_role == "coder"
                                        and coder_system is not None
                                    )
                                    else system
                                ),
                                original_user=user,
                                path=path,
                            )
                        except RawGenerationError as exc:
                            self.last_model_failures += 1
                            final_error = str(exc)
                            self._model_log(candidate, final_error, kind="raw_failure")
                            break

                        action = {
                            "tool": "append_file_raw",
                            "args": {
                                "path": path,
                                "_raw_content": raw_content,
                            },
                        }

                        validate_action(action)

                        if raw_role == "coder":
                            self._record_model_success("coder")
                            self._record_model_success(candidate)
                            return "coder", action

                    self._record_model_success(candidate)
                    return candidate, action
                except Exception as exc:
                    self.last_model_failures += 1
                    final_error = str(exc)
                    self._record_model_failure(candidate)

                    if repair_index < self.config.model_repair_attempts:
                        attempt = repair_index + 1
                        self._model_log(
                            candidate,
                            f"self-repair {attempt}/{self.config.model_repair_attempts}: {exc}",
                            kind="model_repair",
                        )
                        request_user = build_model_repair_prompt(
                            original_user=user,
                            raw_response=raw,
                            error=str(exc),
                            attempt=attempt,
                            max_attempts=self.config.model_repair_attempts,
                        )
                        continue

                    self._model_log(
                        candidate,
                        f"unusable after {self.config.model_repair_attempts} repair attempt(s): {exc}",
                        kind="model_failure",
                    )

            errors.append(f"{candidate}: {final_error or 'unknown model failure'}")

        raise HelixError(
            "No model produced a valid action: " + " | ".join(errors)
        )


TOOLS = {
    "list_files": {"args": {}},
    "read_file": {
        "args": {
            "path": "string",
            "start_line": "int?",
            "end_line": "int?"
        }
    },
    "search_text": {"args": {"query": "string"}},
    "hash_file": {"args": {"path": "string"}},
    "begin_file": {"args": {"path": "string"}},
    "append_file": {
        "args": {
            "path": "string",
            "content": "string <= recommended chunk size"
        }
    },
    "append_file_raw": {
        "args": {
            "path": "string"
        }
    },
    "finish_file": {"args": {"path": "string"}},
    "patch_file": {
        "args": {
            "path": "string",
            "op": "replace | insert_before | insert_after",
            "old": "string?",
            "new": "string?",
            "anchor": "string?",
            "content": "string?",
            "expected_matches": "int?"
        }
    },
    "run_command": {
        "args": {
            "command": "string",
            "timeout_seconds": "int?",
            "verification": "bool?"
        }
    },
    "git_diff": {"args": {}},
    "finish": {"args": {"summary": "string"}}
}


RAW_CONTENT_START = "<<<HELIX_CONTENT"
RAW_CONTENT_END = "HELIX_CONTENT"


def extract_raw_action(text: str) -> dict[str, Any] | None:
    if RAW_CONTENT_START not in text:
        return None

    start = text.find(RAW_CONTENT_START)
    header_text = text[:start].strip()

    if not header_text:
        raise HelixError(
            "Raw Helix response is missing the JSON action header before "
            "<<<HELIX_CONTENT."
        )

    try:
        action = json.loads(header_text)
    except json.JSONDecodeError as exc:
        raise HelixError(
            f"Invalid raw-action JSON header: {exc}. "
            f"Header: {header_text[:500]}"
        ) from exc

    if not isinstance(action, dict):
        raise HelixError("Raw-action header must be a JSON object.")

    if action.get("tool") != "append_file_raw":
        raise HelixError(
            "<<<HELIX_CONTENT may only be used with append_file_raw."
        )

    body_start = start + len(RAW_CONTENT_START)

    if text[body_start:body_start + 2] == "\r\n":
        body_start += 2
    elif text[body_start:body_start + 1] == "\n":
        body_start += 1

    end_marker = "\n" + RAW_CONTENT_END
    end = text.rfind(end_marker)

    if end < body_start:
        raise HelixError(
            "Raw append_file_raw response is missing the closing "
            "HELIX_CONTENT marker on its own line."
        )

    trailing = text[end + len(end_marker):].strip()
    if trailing:
        raise HelixError(
            "Unexpected text after HELIX_CONTENT. "
            "The raw file action must end immediately after the closing marker."
        )

    raw_content = text[body_start:end] + "\n"

    args = action.get("args")
    if not isinstance(args, dict):
        raise HelixError(
            'append_file_raw header requires an "args" JSON object.'
        )

    args = dict(args)
    args["_raw_content"] = raw_content
    action = dict(action)
    action["args"] = args

    return action


def infer_attempted_tool(raw_response: str) -> str | None:
    match = re.search(
        r'["\']tool["\']\s*:\s*["\']([^"\']+)["\']',
        raw_response,
    )
    if not match:
        return None
    return match.group(1)


def repair_schema_guidance(raw_response: str) -> str:
    attempted = infer_attempted_tool(raw_response)

    if attempted in {"append_file", "append_file_raw"}:
        return (
            "For source-code appends, use append_file_raw. "
            "Return ONLY this small JSON action header:\n\n"
            '{"tool":"append_file_raw","args":{"path":"PATH"}}\n\n'
            "Do not include source code in this JSON response. "
            "After Helix accepts the action header, Helix will make a separate "
            "raw-content request where JSON mode is disabled."
        )

    if attempted and attempted in TOOLS:
        return (
            "VALID TOOL SCHEMA:\n"
            + json.dumps(
                {
                    "tool": attempted,
                    "args": TOOLS[attempted]["args"],
                },
                indent=2,
            )
        )

    return (
        "AVAILABLE TOOL SCHEMAS:\n"
        + json.dumps(TOOLS, indent=2)
    )


def build_escalation_handoff(
    *,
    original_user: str,
    previous_errors: list[str],
) -> str:
    try:
        state = json.loads(original_user)
    except json.JSONDecodeError:
        state = {}

    if not isinstance(state, dict):
        state = {}

    task = state.get("task", "")
    staged_files = state.get("staged_files", [])
    active_staged_file = state.get("active_staged_file")
    changed_files = state.get("changed_files", [])
    last_completed_file = state.get("last_completed_file", "")
    post_finish_pending = bool(
        state.get("post_finish_pending")
    )
    remaining_requirements = str(
        state.get("remaining_requirements_instruction", "")
    )
    last_tool_failure = state.get("last_tool_failure")
    verification_passed = bool(state.get("verification_passed"))
    recent_history = state.get("recent_history", [])

    if active_staged_file:
        recommended = (
            f"Continue the active staged file {active_staged_file}. "
            "Prefer append_file_raw for source code. "
            "Do not rescan or switch files until it is finished."
        )
    elif last_tool_failure:
        recommended = (
            "Recover from the recorded tool failure using its exact diagnostics. "
            "Do not repeat the failed action unchanged."
        )
    elif post_finish_pending:
        recommended = (
            remaining_requirements
            or (
                "The previous file was successfully completed. "
                "Advance to the next missing requirement from the original task. "
                "Do not rescan or reread completed work without a diagnostic reason."
            )
        )
    elif not verification_passed:
        recommended = (
            "Continue the original task from the current repository state. "
            "Determine which requirements remain, avoid repeating completed work, "
            "and reach the requested final verification."
        )
    else:
        recommended = (
            "Verification has passed. Confirm all original requirements are "
            "satisfied and finish the task without repeating completed work."
        )

    recent = recent_history[-8:] if isinstance(recent_history, list) else []

    handoff = {
        "original_task": task,
        "changed_or_completed_files": changed_files,
        "active_staged_file": active_staged_file,
        "staged_files": staged_files,
        "last_completed_file": last_completed_file,
        "post_finish_pending": post_finish_pending,
        "remaining_requirements": remaining_requirements,
        "last_tool_failure": last_tool_failure,
        "verification_passed": verification_passed,
        "recent_actions": recent,
        "previous_model_errors": previous_errors[-4:],
        "recommended_next_action": recommended,
    }

    return (
        "ESCALATION HANDOFF - LARGE MODEL\n\n"
        "You are continuing work already performed by another model. "
        "Do not restart the task. Do not assume the repository is empty.\n\n"
        "IMPORTANT: This handoff contains shared Helix task state and tool "
        "results, not the previous model's private reasoning.\n\n"
        + json.dumps(handoff, indent=2, ensure_ascii=False)
        + "\n\nCURRENT SHARED HELIX STATE:\n"
        + original_user
        + "\n\nContinue from this exact state."
    )


def extract_action(text: str) -> dict[str, Any]:
    original = text

    raw_action = extract_raw_action(original)
    if raw_action is not None:
        return raw_action

    text = text.strip()

    if not text:
        raise HelixError("Model returned an empty response.")

    direct_error: json.JSONDecodeError | None = None

    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise HelixError("Model action must be a JSON object.")
        return value
    except json.JSONDecodeError as exc:
        direct_error = exc

    match = re.search(r"{.*}", text, flags=re.DOTALL)
    if not match:
        if text.lstrip().startswith("{") and direct_error is not None:
            raise HelixError(
                f"Invalid model JSON: {direct_error}. "
                f"Response: {original[:500]}"
            ) from direct_error

        raise HelixError(
            f"Model did not return a JSON object. "
            f"Response: {original[:500]}"
        )

    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise HelixError(
            f"Invalid model JSON: {exc}. Response: {original[:500]}"
        ) from exc

    if not isinstance(value, dict):
        raise HelixError("Model action must be a JSON object.")

    return value


def validate_action(action: dict[str, Any]) -> None:
    if not isinstance(action, dict):
        raise HelixError("Model action must be a JSON object.")

    tool = action.get("tool")
    if not isinstance(tool, str) or not tool:
        raise HelixError(
            'Model action is missing required string field "tool".'
        )

    if tool not in TOOLS:
        allowed = ", ".join(sorted(TOOLS))
        raise HelixError(
            f'Unknown Helix tool "{tool}". Allowed tools: {allowed}.'
        )

    if "args" not in action:
        raise HelixError(
            f'Model action for "{tool}" is missing required object field "args".'
        )

    args = action["args"]
    if not isinstance(args, dict):
        raise HelixError(
            f'Model action "{tool}".args must be a JSON object.'
        )

    expected = TOOLS[tool]["args"]
    allowed_arguments = set(expected)

    if tool == "append_file_raw":
        allowed_arguments.add("_raw_content")

    unexpected = sorted(set(args) - allowed_arguments)
    if unexpected:
        raise HelixError(
            f'Model action "{tool}" has unexpected argument(s): '
            + ", ".join(unexpected)
            + "."
        )

    for name, type_spec in expected.items():
        optional = str(type_spec).endswith("?")

        if name not in args:
            if optional:
                continue
            raise HelixError(
                f'Model action "{tool}" is missing required '
                f'argument "{name}". Required shape: '
                + json.dumps({"tool": tool, "args": expected})
            )

        value = args[name]
        if value is None and optional:
            continue

        base_type = str(type_spec).rstrip("?")

        if base_type.startswith("string") or " | " in base_type:
            valid = isinstance(value, str)
        elif base_type.startswith("int"):
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif base_type.startswith("bool"):
            valid = isinstance(value, bool)
        else:
            valid = True

        if not valid:
            raise HelixError(
                f'Model action "{tool}" argument "{name}" '
                f'must match {type_spec}; received '
                f'{type(value).__name__}.'
            )


def build_model_repair_prompt(
    *,
    original_user: str,
    raw_response: str,
    error: str,
    attempt: int,
    max_attempts: int,
) -> str:
    response_excerpt = raw_response[:6000]
    schema_guidance = repair_schema_guidance(raw_response)

    return (
        "MODEL PROTOCOL REPAIR REQUIRED\n\n"
        "Your previous Helix action was invalid. Do not repeat it unchanged.\n\n"
        f"REPAIR ATTEMPT: {attempt}/{max_attempts}\n\n"
        f"EXACT ERROR:\n{error}\n\n"
        "PREVIOUS RESPONSE:\n"
        f"{response_excerpt if response_excerpt.strip() else '[EMPTY RESPONSE]'}\n\n"
        "CURRENT TASK STATE AND STAGED-WORK CONTEXT:\n"
        f"{original_user}\n\n"
        f"{schema_guidance}\n\n"
        "Return exactly ONE corrected JSON object and nothing else.\n"
        "For append_file_raw, return only the small JSON header with path. "
        "Helix performs raw source generation in a second non-JSON model call.\n"
        'Required top-level shape: {"tool":"TOOL_NAME","args":{...}}\n'
        "Do not output markdown, analysis, apologies, or commentary.\n"
        "Preserve the current task state.\n"
        "If staged_files is non-empty, continue the currently staged work "
        "with append_file, read_file, or finish_file as appropriate.\n"
        "Do not restart an already staged file.\n"
        "Do not rescan the repository merely because the previous model "
        "response was malformed.\n"
        "Fix the exact protocol error above and continue the task."
    )

class SessionStore:
    def __init__(self, project_root: Path):
        self.path = project_root / ".helix" / "history.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, task: str, status: str, summary: str, steps: int, duration_seconds: float) -> None:
        entry = {
            "time": int(time.time()),
            "task": task,
            "status": status,
            "summary": summary,
            "steps": steps,
            "duration_seconds": round(duration_seconds, 2),
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                entries.append(value)
        return entries[-limit:]


class HelixConsole:
    def __init__(self, config: HelixConfig, project_root: Path, permissions: AgentPermissions):
        self.config = config
        self.project_root = project_root
        self.permissions = permissions

    @staticmethod
    def model_label(spec: ModelSpec | None) -> str:
        if not spec or not spec.enabled or not spec.model:
            return "disabled"
        return f"{spec.provider}/{spec.model}"

    def banner(self) -> None:
        print()
        print("  +--------------------------------------------------------------+")
        print("  |                         H E L I X                            |")
        print("  |                  Autonomous Coding Agent                    |")
        print("  +--------------------------------------------------------------+")
        print(f"  Version   {VERSION}")
        print(f"  Project   {self.project_root}")
        if (
            "architect" in self.config.models
            and "orchestrator" in self.config.models
            and "coder" in self.config.models
        ):
            print(
                f"  Architect    "
                f"{self.model_label(self.config.models.get('architect'))}"
            )
            print(
                f"  Orchestrator "
                f"{self.model_label(self.config.models.get('orchestrator'))}"
            )
            print(
                f"  Coder        "
                f"{self.model_label(self.config.models.get('coder'))}"
            )
            print(
                f"  Remote       "
                f"{self.model_label(self.config.models.get('remote'))}"
            )
        else:
            print(f"  Main      {self.model_label(self.config.models.get('main'))}")
            print(f"  Large     {self.model_label(self.config.models.get('large'))}")
            print(f"  Remote    {self.model_label(self.config.models.get('remote'))}")
        print(f"  Forge     {'enabled' if self.config.patchforge_cli else 'not configured'}")
        print(f"  Write     {'allowed' if self.permissions.write else 'READ ONLY'}")
        print(f"  Execute   {'allowed' if self.permissions.execute else 'DISABLED'}")
        print(f"  Steps     {'unlimited' if self.config.max_steps == 0 else self.config.max_steps}")
        print(
            f"  Stall     escalate={self.config.stall_escalation_steps} "
            f"stop={self.config.stall_stop_steps}"
        )
        print("  " + "-" * 64)

    def task_start(self, task: str) -> None:
        print()
        first_line = next(
            (line.strip() for line in task.splitlines() if line.strip()),
            "Task",
        )
        suffix = " [multiline]" if "\n" in task else ""
        print(f"  TASK      {first_line}{suffix}")
        print("  " + "-" * 64)

    def step(self, step: int, role: str, tool: str) -> None:
        labels = {
            "list_files": "SCAN",
            "read_file": "READ",
            "search_text": "SEARCH",
            "hash_file": "HASH",
            "begin_file": "BEGIN",
            "append_file": "WRITE",
            "append_file_raw": "WRITE",
            "finish_file": "CREATE",
            "patch_file": "FORGE",
            "run_command": "EXEC",
            "git_diff": "DIFF",
            "finish": "DONE",
        }
        print(f"  [{step:03d}] {role.upper():6} {labels.get(tool, tool.upper()):7} {tool}")

    def result(self, ok: bool, summary: str) -> None:
        marker = "PASS" if ok else "FAIL"
        text = summary.strip()
        if len(text) > 1200:
            text = text[:1200] + "\n... output truncated ..."
        if "\n" not in text:
            print(f"        {marker}  {text}")
            return
        print(f"        {marker}")
        for line in text.splitlines():
            print(f"        | {line}")

    def model_error(self, step: int, message: str) -> None:
        print(f"  [{step:03d}] MODEL  ERROR  {message}")

    def stall(self, step: int, message: str) -> None:
        print(f"  [{step:03d}] LOOP   STALL  {message}")

    def escalation(self, step: int, role: str, stall_steps: int) -> None:
        alternate_available = any(
            (
                spec
                and spec.enabled
                and spec.model
            )
            for name, spec in self.config.models.items()
            if name in {"large", "remote"}
        )

        if role == "main" and not alternate_available:
            print(
                f"  [{step:03d}] ROUTER HOLD -> MAIN "
                f"(stall score {stall_steps}; no fallback available)"
            )
            return

        print(
            f"  [{step:03d}] ROUTER ESCALATE -> {role.upper()} "
            f"(stall score {stall_steps})"
        )

    def complete(self, summary: str, duration_seconds: float) -> None:
        print("  " + "-" * 64)
        print(f"  DONE      {summary}")
        print(f"  Duration  {duration_seconds:.2f}s")
        print()

    def stopped(self, message: str) -> None:
        print("  " + "-" * 64)
        print(f"  STOP      {message}")
        print()

    def show_models(self) -> None:
        roles = (
            ("architect", "orchestrator", "coder", "remote")
            if "architect" in self.config.models
            else ("small", "main", "large", "remote")
        )

        for role in roles:
            print(
                f"  {role:12} "
                f"{self.model_label(self.config.models.get(role))}"
            )

    @staticmethod
    def show_history(entries: list[dict[str, Any]]) -> None:
        if not entries:
            print("  No previous Helix tasks for this project.")
            return
        for entry in entries:
            print(f"  {str(entry.get('status', '?')).upper():8} {entry.get('steps', '?')} steps   {entry.get('task', '')}")



class HelixAgent:
    def __init__(self, project_root: Path, config: HelixConfig, prompt_path: Path, permissions: AgentPermissions | None = None):
        self.project_root = project_root.resolve()
        self.config = config
        self.permissions = permissions or AgentPermissions()
        self.workspace = Workspace(self.project_root)
        self.runner = CommandRunner(self.project_root, config.command_timeout_seconds)
        self.forge = PatchForgeAdapter(self.project_root, config.patchforge_cli)
        self.router = ModelRouter(config)
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

        prompt_root = Path(__file__).parent / "prompts"

        self.coder_system_prompt = (
            prompt_root / "code_writer.txt"
        ).read_text(encoding="utf-8")

        self.architect_system_prompt = (
            prompt_root / "architect.txt"
        ).read_text(encoding="utf-8")

        self.console = HelixConsole(config, self.project_root, self.permissions)
        self.sessions = SessionStore(self.project_root)

    @staticmethod
    def _action_signature(tool: str, args: dict[str, Any]) -> str:
        return tool + ":" + json.dumps(
            args,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _failure_signature(
        tool: str,
        args: dict[str, Any],
        summary: str,
    ) -> str:
        normalized = re.sub(
            r'"duration_seconds":\s*[0-9.]+',
            '"duration_seconds":<time>',
            summary,
        )
        return (
            HelixAgent._action_signature(tool, args)
            + ":"
            + normalized[:3000]
        )

    @staticmethod
    def _duplicate_sensitive(tool: str) -> bool:
        return tool in {
            "list_files",
            "read_file",
            "search_text",
            "hash_file",
            "git_diff",
        }

    def _staged_work_instruction(self, state: AgentState) -> str:
        if not state.staged_files:
            return ""

        items = sorted(state.staged_files.items())
        details = ", ".join(
            f"{path} ({characters} characters)"
            for path, characters in items
        )
        first_path = items[0][0]

        return (
            f"Unfinished staged files: {details}. "
            f"Continue staged work before rescanning. "
            f"For {first_path}, prefer append_file_raw to add remaining source "
            "or finish_file if the file is complete. "
            "Do not call begin_file again for an already staged file."
        )

    def _remaining_requirements_instruction(
        self,
        state: AgentState,
    ) -> str:
        task = state.task.lower()
        changed = {
            path.lower()
            for path in state.changed_files
        }

        remaining: list[str] = []

        test_requested = any(
            token in task
            for token in (
                "test",
                "tests",
                "pytest",
                "unittest",
            )
        )

        has_test_file = any(
            (
                Path(path).name.startswith("test_")
                or Path(path).name.endswith("_test.py")
                or "/tests/" in path.replace("\\", "/")
                or path.replace("\\", "/").startswith("tests/")
            )
            for path in changed
        )

        if test_requested and not has_test_file:
            remaining.append(
                "Automated tests are requested by the original task, "
                "but no test-like file has been completed yet."
            )

        if not state.verification_passed:
            remaining.append(
                "Final verification has not passed yet."
            )

        if state.last_completed_file:
            remaining.append(
                f"{state.last_completed_file} was just completed. "
                "Treat it as completed work unless a real diagnostic "
                "requires changing it."
            )

        if not remaining:
            remaining.append(
                "Review the original task for any remaining requirement "
                "and proceed directly to completion."
            )

        return (
            "POST-FINISH CONTINUATION REQUIRED. "
            + " ".join(remaining)
            + " Do not call list_files merely to rediscover repository state. "
            "Choose the next concrete implementation, testing, debugging, "
            "or verification action."
        )

    @staticmethod
    def _tool_entry_name(entry: Any) -> str:
        """Return the tool name from Helix's prompt/tool schema entry."""
        if isinstance(entry, str):
            return entry

        if not isinstance(entry, dict):
            return ""

        for key in ("name", "tool"):
            value = entry.get(key)
            if isinstance(value, str):
                return value

        function = entry.get("function")
        if isinstance(function, dict):
            value = function.get("name")
            if isinstance(value, str):
                return value

        return ""

    def _available_tools_for_state(
        self,
        state: AgentState,
    ) -> list[Any]:
        """Expose only tools that are legal/useful in the current state."""
        active = self._active_staged_file(state)

        if active:
            characters = state.staged_files.get(active, 0)

            if characters == 0:
                allowed = {
                    "append_file",
                    "append_file_raw",
                }
            else:
                allowed = {
                    "append_file",
                    "append_file_raw",
                    "read_file",
                    "finish_file",
                }

            return [
                entry
                for entry in TOOLS
                if self._tool_entry_name(entry) in allowed
            ]

        if state.post_finish_pending:
            # If the original task still requires an automated test file,
            # force the transition into that concrete next file instead of
            # allowing repository scans, unrelated reads, or commands.
            if self._post_finish_requires_test_file(state):
                return [
                    entry
                    for entry in TOOLS
                    if self._tool_entry_name(entry) == "begin_file"
                ]

            verification_command = (
                self._post_finish_verification_command(state)
            )

            if verification_command:
                return [
                    entry
                    for entry in TOOLS
                    if self._tool_entry_name(entry) == "run_command"
                ]

            # Otherwise the previous file is complete. Avoid pointless
            # rediscovery/staged operations while allowing concrete follow-up
            # implementation, patching, or verification actions.
            blocked = {
                "list_files",
                "read_file",
                "append_file",
                "append_file_raw",
                "finish_file",
            }

            filtered = [
                entry
                for entry in TOOLS
                if self._tool_entry_name(entry) not in blocked
            ]

            return filtered or TOOLS

        return TOOLS

    def _repair_staged_python_file(
        self,
        relative: str,
        current_content: str,
        validation_error: str,
        state: AgentState,
    ) -> tuple[str | None, str]:
        """Repair an invalid staged Python file as a complete file."""

        repair_role = (
            "coder"
            if self.router._new_role_architecture()
            else "main"
        )
        backend = self.router.backend(repair_role)
        error = validation_error
        previous = current_content

        is_test_file = (
            Path(relative).name.lower().startswith("test_")
            or Path(relative).name.lower().endswith("_test.py")
        )

        for attempt in range(1, 3):
            print(
                f"[CORE] staged validation repair "
                f"{attempt}/2 -> {relative}"
            )

            test_instruction = ""

            if is_test_file:
                test_instruction = (
                    "\nThis is a pytest test file. It MUST contain real "
                    "pytest-discoverable test functions whose names begin "
                    "with test_. Do not rewrite the application inside the "
                    "test file. Test the implementation file instead.\n"
                )

            context = {
                "path": relative,
                "validation_error": error,
                "last_completed_file": state.last_completed_file,
                "recent_history": state.recent_context()[-6:],
            }

            prompt = (
                "HELIX STAGED PYTHON FULL-FILE REPAIR\n\n"
                f"Repair the complete staged Python file: {relative}\n"
                "Return ONLY the complete corrected file contents.\n"
                "Do NOT return JSON.\n"
                "Do NOT use markdown fences.\n"
                "Do NOT explain anything.\n"
                "Do NOT return a patch or diff.\n"
                "Do NOT return only a continuation.\n"
                "Preserve the intended functionality while correcting the "
                "validation failure.\n"
                + test_instruction
                + "\nREPAIR CONTEXT:\n"
                + json.dumps(
                    context,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n\nCURRENT STAGED FILE:\n"
                + previous
            )

            try:
                raw = interruptible_call(
                    lambda: backend.complete(
                        (
                            self.coder_system_prompt
                            if repair_role == "coder"
                            else self.system_prompt
                        ),
                        prompt,
                        json_mode=False,
                    )
                )
            except Exception as exc:
                error = f"Repair model request failed: {exc}"
                continue

            try:
                repaired, unwrapped = (
                    self.router._unwrap_single_markdown_fence(raw)
                )
            except Exception as exc:
                error = str(exc)
                continue

            if not repaired.strip():
                error = "Repair model returned an empty file."
                continue

            if "<<<HELIX_CONTENT" in repaired:
                error = (
                    "Repair model returned HELIX_CONTENT markers."
                )
                continue

            if not repaired.endswith("\n"):
                repaired += "\n"

            repaired_error = self._staged_python_validation_error(
                relative,
                repaired,
            )

            if repaired_error:
                error = repaired_error
                previous = repaired
                continue

            print(
                f"[CORE] staged validation PASS -> {relative}"
            )

            return repaired, ""

        return None, error

    @staticmethod
    def _staged_python_validation_error(
        relative: str,
        content: str,
    ) -> str | None:
        """Validate staged Python before Patch Forge is allowed to create it."""
        path = Path(relative)

        if path.suffix.lower() != ".py":
            return None

        try:
            compile(
                content,
                relative,
                "exec",
            )
        except SyntaxError as exc:
            line = exc.lineno or "?"
            message = exc.msg or "invalid syntax"
            source = (exc.text or "").strip()

            detail = (
                f" Source: {source}"
                if source
                else ""
            )

            return (
                f"Python syntax validation failed for {relative} "
                f"at line {line}: {message}.{detail}"
            )

        filename = path.name.lower()
        is_test_file = (
            filename.startswith("test_")
            or filename.endswith("_test.py")
        )

        if is_test_file:
            discoverable_test = re.search(
                r"(?m)^\s*def\s+test_[A-Za-z0-9_]*\s*\(",
                content,
            )

            if not discoverable_test:
                return (
                    f"Pytest discovery validation failed for {relative}: "
                    "the file contains no discoverable test_* function. "
                    "Add real pytest tests before finishing this file."
                )

        return None

    @staticmethod
    def _post_finish_verification_command(
        state: AgentState,
    ) -> str:
        """Return an explicit final verification requested by the task."""
        if (
            not state.post_finish_pending
            or state.verification_passed
        ):
            return ""

        task = state.task.lower()

        if "pytest -q" in task:
            return "pytest -q"

        if "pytest" in task:
            return "pytest -q"

        return ""

    def _post_finish_requires_test_file(
        self,
        state: AgentState,
    ) -> bool:
        if not state.post_finish_pending:
            return False

        guidance = self._remaining_requirements_instruction(
            state
        ).lower()

        return (
            "automated tests are requested" in guidance
            and "no test-like file has been completed yet" in guidance
        )

    @staticmethod
    def _post_finish_test_path(state: AgentState) -> str:
        completed = str(
            state.last_completed_file or "app.py"
        )

        path = Path(completed)
        stem = path.stem

        if stem.startswith("test_"):
            stem = stem[5:] or "app"

        return str(
            path.with_name(f"test_{stem}.py")
        ).replace("\\", "/")

    def _post_finish_action_lock_error(
        self,
        tool: str,
        args: dict[str, Any],
        state: AgentState,
    ) -> str | None:
        if not state.post_finish_pending:
            return None

        if self._post_finish_requires_test_file(state):
            expected = self._post_finish_test_path(state)

            if tool != "begin_file":
                return (
                    "POST-FINISH ACTION LOCK: automated tests are still "
                    f"required. The next action must be begin_file for "
                    f"{expected}. Tool \"{tool}\" is unavailable until "
                    "the required test file is started."
                )

            target = str(args.get("path", ""))

            try:
                normalized = self.workspace.normalize_relative(
                    target
                )
            except Exception:
                normalized = target.replace("\\", "/")

            if normalized != expected:
                return (
                    "POST-FINISH ACTION LOCK: automated tests are still "
                    f"required. Begin the expected test file {expected}, "
                    f'not "{target}".'
                )

            return None

        verification_command = (
            self._post_finish_verification_command(state)
        )

        if verification_command:
            proposed_command = str(
                args.get("command", "")
            ).strip()

            if (
                tool != "run_command"
                or proposed_command != verification_command
                or not bool(args.get("verification"))
            ):
                return (
                    "POST-FINISH ACTION LOCK: final verification is now "
                    f"required. Run exactly {verification_command!r} using "
                    "run_command with verification=true. Do not scan or "
                    "perform unrelated work first."
                )

            return None

        # Reuse the existing post-finish safety rules, but run them before
        # tool execution so invalid actions do not consume agent steps.
        return self._post_finish_guard(
            tool,
            args,
            state,
        )

    def _post_finish_guard(
        self,
        tool: str,
        args: dict[str, Any],
        state: AgentState,
    ) -> str | None:
        if not state.post_finish_pending:
            return None

        completed = state.last_completed_file
        guidance = self._remaining_requirements_instruction(state)

        if tool == "list_files":
            return (
                f"Post-finish repository scan blocked. {completed or 'The previous file'} "
                f"was already completed. {guidance}"
            )

        if tool == "read_file" and completed:
            target = str(args.get("path", ""))

            try:
                target = self.workspace.normalize_relative(target)
            except Exception:
                pass

            if target == completed:
                return (
                    f"Immediate reread of completed file {completed} blocked. "
                    "There is no diagnostic failure requiring that reread. "
                    + guidance
                )

        return None

    def _post_finish_instruction(self, state: AgentState) -> str:
        if not state.last_completed_file:
            return ""

        return self._remaining_requirements_instruction(state)

    def _active_staged_file(self, state: AgentState) -> str | None:
        if not state.staged_files:
            return None

        # File creation is intentionally serialized. If multiple staged files
        # somehow exist, continue the oldest stable ordering rather than
        # allowing the model to wander into unrelated files.
        return sorted(state.staged_files)[0]

    def _staged_action_lock_error(
        self,
        tool: str,
        args: dict[str, Any],
        state: AgentState,
    ) -> str | None:
        active = self._active_staged_file(state)
        if not active:
            return None

        characters = state.staged_files.get(active, 0)

        if characters == 0:
            allowed = {"append_file_raw", "append_file"}
        else:
            allowed = {
                "append_file_raw",
                "append_file",
                "read_file",
                "finish_file",
            }

        if tool not in allowed:
            choices = ", ".join(sorted(allowed))
            return (
                f"STAGED WORK LOCK: {active} is unfinished "
                f"({characters} characters staged). "
                f'Tool "{tool}" is unavailable right now. '
                f"Allowed tools: {choices}."
            )

        target = str(args.get("path", ""))
        if not target:
            return (
                f"STAGED WORK LOCK: tool {tool} must target "
                f'the active staged file "{active}".'
            )

        try:
            normalized = self.workspace.normalize_relative(target)
        except Exception:
            normalized = target

        if normalized != active:
            return (
                f"STAGED WORK LOCK: {active} is the only file that may be "
                f"worked on right now. The proposed action targeted "
                f'"{target}". Continue {active} instead.'
            )

        return None

    def _staged_work_guard(
        self,
        tool: str,
        args: dict[str, Any],
        state: AgentState,
    ) -> str | None:
        active = self._active_staged_file(state)
        if not active:
            return None

        target = str(args.get("path", ""))

        if tool == "list_files":
            return (
                f"Unfinished staged file: {active}. "
                "Repository rescanning is blocked while staged work is active. "
                f"Continue {active} using append_file, read_file, or finish_file."
            )

        if tool == "read_file" and target:
            try:
                normalized = self.workspace.normalize_relative(target)
            except Exception:
                normalized = target

            if normalized != active:
                return (
                    f"Unrelated read blocked. {active} is currently staged and "
                    "unfinished. "
                    f'Do not read "{target}". '
                    f'Continue using append_file(path="{active}"), '
                    f'read_file(path="{active}"), or '
                    f'finish_file(path="{active}").'
                )

        if tool == "begin_file" and target:
            try:
                normalized = self.workspace.normalize_relative(target)
            except Exception:
                normalized = target

            if normalized != active:
                return (
                    f"Cannot begin {target} while {active} is staged and unfinished. "
                    f"Complete {active} first."
                )

        return None

    def _tool_schema_for_repair(
        self,
        attempted_action: dict[str, Any] | None,
    ) -> str:
        if not attempted_action:
            return json.dumps(TOOLS, indent=2)

        tool = attempted_action.get("tool")
        if isinstance(tool, str) and tool in TOOLS:
            return json.dumps(
                {
                    "tool": tool,
                    "args": TOOLS[tool],
                },
                indent=2,
            )

        return json.dumps(TOOLS, indent=2)

    def _failure_recovery_instruction(self, state: AgentState) -> str:
        failure = state.last_tool_failure
        if not failure:
            return ""

        tool = str(failure.get("tool", ""))
        path = str(failure.get("path", ""))

        if tool == "patch_file":
            active = self._active_staged_file(state)
            staged_note = (
                f" Active staged file: {active}. Finish that staged file before "
                "switching to unrelated work."
                if active
                else ""
            )

            return (
                "PATCH RECOVERY REQUIRED. "
                f"The previous patch_file for {path or 'the target file'} failed. "
                "Do not call list_files and do not repeat the same patch unchanged. "
                "Inspect the failed file with read_file and use the exact failure "
                "details to construct a corrected patch_file action. "
                "Resolve this patch failure before unrelated repository scanning."
                + staged_note
            )

        return (
            "TOOL RECOVERY REQUIRED. Inspect the previous failed action and "
            "correct it before unrelated repository scanning."
        )

    def _mark_progress(self, state: AgentState) -> None:
        state.stall_steps = 0
        state.stall_level = 0
        state.last_failure_signature = ""

    def _mark_stall(self, state: AgentState, reason: str) -> bool:
        state.stall_steps += 1
        self.console.stall(state.step, reason)

        escalation = self.config.stall_escalation_steps
        new_level = state.stall_level

        if state.stall_steps >= escalation:
            new_level = max(new_level, 1)

        if (
            state.stall_steps >= escalation * 2
            and self.router._enabled("remote")
        ):
            new_level = max(new_level, 2)

        if new_level > state.stall_level:
            state.stall_level = new_level
            role = self.router.choose_role(state)
            self.console.escalation(
                state.step,
                role,
                state.stall_steps,
            )

        return state.stall_steps >= self.config.stall_stop_steps

    def _note_failure(
        self,
        state: AgentState,
        tool: str,
        args: dict[str, Any],
        summary: str,
    ) -> bool:
        signature = self._failure_signature(tool, args, summary)

        if signature == state.last_failure_signature:
            return self._mark_stall(
                state,
                f"Repeated unchanged failure from {tool}.",
            )

        state.last_failure_signature = signature
        return False

    def _stop_task(
        self,
        state: AgentState,
        started: float,
        message: str,
    ) -> int:
        duration = time.monotonic() - started
        self.sessions.record(
            task=state.task,
            status="stopped",
            summary=message,
            steps=state.step,
            duration_seconds=duration,
        )
        self.console.stopped(message)
        return 2

    @staticmethod
    def _extract_architect_contract(plan: str) -> str:
        start_marker = (
            "<!-- HELIX_EXECUTION_CONTRACT_START -->"
        )
        end_marker = (
            "<!-- HELIX_EXECUTION_CONTRACT_END -->"
        )

        start = plan.find(start_marker)
        end = plan.find(end_marker)

        if (
            start >= 0
            and end > start
        ):
            start += len(start_marker)
            contract = plan[start:end].strip()

            if contract:
                return contract[:5000]

        # A missing marker should not throw away a useful plan.
        return plan[:5000].strip()

    def _architect_enabled(self) -> bool:
        spec = self.config.models.get("architect")
        return bool(
            spec
            and spec.enabled
            and spec.model
        )

    def _architect_request(
        self,
        prompt: str,
    ) -> str:
        if not self._architect_enabled():
            raise HelixError(
                "Architect model is not configured."
            )

        self.router.unload_roles(
            ("orchestrator", "coder", "remote")
        )

        backend = self.router.backend("architect")

        max_empty_attempts = 2
        max_continuations = 5
        max_plan_chars = 60000

        parts: list[str] = []
        empty_attempt = 0
        continuation_count = 0
        current_prompt = prompt
        last_diagnostic = ""

        try:
            while True:
                started = time.monotonic()
                stop_heartbeat = threading.Event()

                def heartbeat() -> None:
                    while not stop_heartbeat.wait(10.0):
                        elapsed = (
                            time.monotonic()
                            - started
                        )

                        print(
                            "[ARCHITECT] "
                            f"{elapsed:.0f}s elapsed | "
                            "waiting for model..."
                        )

                heartbeat_thread = threading.Thread(
                    target=heartbeat,
                    name="helix-architect-heartbeat",
                    daemon=True,
                )

                heartbeat_thread.start()

                try:
                    if continuation_count:
                        label = (
                            f"continuation "
                            f"{continuation_count}/{max_continuations}"
                        )
                    else:
                        label = (
                            f"response attempt "
                            f"{empty_attempt + 1}/{max_empty_attempts}"
                        )

                    print(
                        "[ARCHITECT] generating "
                        + label
                        + "..."
                    )

                    response = interruptible_call(
                        lambda: backend.complete(
                            self.architect_system_prompt,
                            current_prompt,
                            json_mode=False,
                        )
                    )
                finally:
                    stop_heartbeat.set()
                    heartbeat_thread.join(
                        timeout=0.25
                    )

                metadata = getattr(
                    backend,
                    "last_response_metadata",
                    {},
                )

                content = response.strip()

                if not content:
                    content_chars = metadata.get(
                        "content_chars",
                        0,
                    )
                    thinking_chars = metadata.get(
                        "thinking_chars",
                        0,
                    )
                    done_reason = metadata.get(
                        "done_reason"
                    )
                    eval_count = metadata.get(
                        "eval_count"
                    )

                    last_diagnostic = (
                        "empty final content; "
                        f"content_chars={content_chars}, "
                        f"thinking_chars={thinking_chars}, "
                        f"eval_count={eval_count}, "
                        f"done_reason={done_reason!r}"
                    )

                    if thinking_chars:
                        print(
                            "[ARCHITECT] model returned "
                            f"{thinking_chars:,} thinking "
                            "characters but no final content."
                        )
                    else:
                        print(
                            "[ARCHITECT] model returned "
                            "no usable final content."
                        )

                    print(
                        "[ARCHITECT] transport metadata: "
                        + last_diagnostic
                    )

                    empty_attempt += 1

                    if empty_attempt < max_empty_attempts:
                        print(
                            "[ARCHITECT] empty-response "
                            "retry 1/1..."
                        )
                        continue

                    raise HelixError(
                        "Architect returned empty final "
                        "content after 2 attempts. "
                        + last_diagnostic
                    )

                empty_attempt = 0
                parts.append(content)

                total_chars = sum(
                    len(part)
                    for part in parts
                )

                done_reason = metadata.get(
                    "done_reason"
                )

                eval_count = metadata.get(
                    "eval_count"
                )

                print(
                    "[ARCHITECT] response received "
                    f"({len(content):,} chars, "
                    f"{total_chars:,} total)"
                )

                details = []

                prompt_eval_count = metadata.get(
                    "prompt_eval_count"
                )

                if prompt_eval_count is not None:
                    details.append(
                        f"prompt={prompt_eval_count} tokens"
                    )

                if eval_count is not None:
                    details.append(
                        f"generated={eval_count} tokens"
                    )

                details.append(
                    f"context={getattr(backend.spec, 'num_ctx', '?')}"
                )

                if done_reason:
                    details.append(
                        f"done={done_reason}"
                    )

                if details:
                    print(
                        "[ARCHITECT] "
                        + ", ".join(details)
                    )

                if total_chars >= max_plan_chars:
                    raise HelixError(
                        "Architect plan exceeded "
                        f"{max_plan_chars:,} characters "
                        "without completing."
                    )

                if done_reason != "length":
                    break

                continuation_count += 1

                if continuation_count > max_continuations:
                    raise HelixError(
                        "Architect exceeded maximum "
                        f"continuations ({max_continuations})."
                    )

                print(
                    "[ARCHITECT] output truncated by "
                    "model length limit; requesting continuation..."
                )

                # Keep only enough preceding output to preserve local
                # continuity. A huge tail wastes the model's context window.
                previous_tail = content[-3000:]

                current_prompt = (
                    "ARCHITECT CONTINUATION REQUIRED\n\n"
                    "Your previous response was cut off because the model "
                    "reached its generation limit.\n\n"
                    "Continue EXACTLY where the previous response stopped.\n"
                    "Do NOT restart the plan.\n"
                    "Do NOT repeat the execution contract.\n"
                    "Do NOT repeat already completed sections.\n"
                    "Do NOT summarize previous content.\n"
                    "Continue the unfinished detailed blueprint until it is "
                    "fully complete.\n\n"
                    "PREVIOUS RESPONSE TAIL:\n"
                    + previous_tail
                )

            combined = "\n".join(
                part.rstrip()
                for part in parts
            ).strip() + "\n"

            print(
                "[ARCHITECT] complete blueprint assembled "
                f"({len(combined):,} chars, "
                f"{len(parts)} part(s))"
            )

            return combined

        finally:
            self.router.unload_role(
                "architect"
            )

    def _create_development_plan(
        self,
        state: AgentState,
    ) -> None:
        if not self._architect_enabled():
            return

        print(
            "[ARCHITECT] pre-coding analysis -> "
            + self.config.models["architect"].model
        )

        inventory = self.workspace.list_files(
            max_files=300
        )

        prompt = (
            "ARCHITECT PRE-CODING PHASE\n\n"
            "Produce an exceptionally detailed, implementation-ready "
            "development blueprint for the task below.\n\n"
            "Do NOT write project source code.\n"
            "Do NOT choose Helix tools.\n"
            "Do NOT claim the application is complete.\n\n"
            "The guide must be detailed enough that a separate small "
            "orchestrator and a separate coding model can implement the "
            "project almost mechanically.\n\n"
            "FIRST, include a concise execution contract between these exact "
            "markers:\n"
            "<!-- HELIX_EXECUTION_CONTRACT_START -->\n"
            "<!-- HELIX_EXECUTION_CONTRACT_END -->\n\n"
            "The execution contract must preserve EVERY explicit user "
            "requirement and exact user-facing terminology. "
            "It may contain only explicit requirements and genuinely "
            "necessary derived correctness requirements. Architecture "
            "choices, optional enhancements, invented package names, "
            "invented paths, invented flags, and invented output formats "
            "must remain outside the execution contract.\n\n"
            "AFTER the contract, first include a REQUIREMENT PROVENANCE "
            "TABLE categorizing every important item as EXPLICIT, DERIVED, "
            "DESIGN, or OPTIONAL, and whether it is binding. Then produce "
            "a very long and exhaustive guide. "
            "Cover every applicable item below:\n\n"
            "1. Exact project objective and definition of success.\n"
            "2. Every explicit user requirement, individually enumerated.\n"
            "3. Implicit requirements necessary for the explicit features "
            "to actually work.\n"
            "4. Exact user-facing commands, names, labels, syntax, APIs, "
            "arguments, outputs, and behavior. Do not silently rename them.\n"
            "5. Complete proposed architecture.\n"
            "6. File and module map with responsibility of every file.\n"
            "7. Data model, schemas, types, persistence format and lifecycle.\n"
            "8. Control flow and data flow.\n"
            "9. Detailed implementation sequence, in dependency order.\n"
            "10. Function/class/interface responsibilities and contracts.\n"
            "11. Input parsing, conversions, validation and normalization.\n"
            "12. Error handling and expected failure behavior.\n"
            "13. Persistence, reload/restart behavior and corruption cases.\n"
            "14. Edge cases and boundary conditions.\n"
            "15. Platform/OS considerations where applicable.\n"
            "16. Dependency strategy.\n"
            "17. Security and safety considerations where applicable.\n"
            "18. Performance considerations where applicable.\n"
            "19. Unit-test strategy.\n"
            "20. Integration-test strategy.\n"
            "21. End-to-end acceptance tests that exercise the REAL "
            "user-facing interface rather than only internal functions.\n"
            "22. Manual smoke-test procedure.\n"
            "23. A requirement-to-test traceability matrix.\n"
            "24. Common implementation mistakes likely to occur.\n"
            "25. Model-specific traps: wrong command names, type conversion, "
            "missing persistence calls, tests that merely agree with a buggy "
            "implementation, incomplete code, duplicated code, and silent "
            "failure paths.\n"
            "26. Debugging strategy for each major subsystem.\n"
            "27. Completion checklist that must be satisfied before Helix "
            "may declare success.\n"
            "28. Any other issue an experienced senior engineer should "
            "anticipate before implementation begins.\n\n"
            "Be concrete and specific. Prefer too much useful engineering "
            "detail over too little.\n\n"
            "ORIGINAL USER TASK:\n"
            + state.task
            + "\n\nCURRENT PROJECT FILE INVENTORY:\n"
            + json.dumps(
                inventory,
                indent=2,
                ensure_ascii=False,
            )
        )

        plan = self._architect_request(prompt)

        state.development_plan = plan
        state.architect_contract = (
            self._extract_architect_contract(plan)
        )

        path = (
            self.project_root
            / ".helix"
            / "development_plan.md"
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            plan,
            encoding="utf-8",
            newline="\n",
        )

        print(
            "[ARCHITECT] development plan saved -> "
            f"{path}"
        )
        print(
            "[ARCHITECT] execution contract "
            f"captured ({len(state.architect_contract):,} chars)"
        )

    def _architect_recovery_package(
        self,
        state: AgentState,
        reason: str,
    ) -> dict[str, Any]:
        active = self._active_staged_file(state)
        staged_source = ""

        if active:
            try:
                staged_source = self.workspace.read_file(
                    active,
                    1,
                    None,
                )
            except Exception:
                staged_source = ""

        return {
            "task": state.task,
            "reason_for_escalation": reason,
            "step": state.step,
            "coding_failures": state.coding_failures,
            "model_failures": state.model_failures,
            "stall_steps": state.stall_steps,
            "workspace_revision": state.workspace_revision,
            "changed_files": sorted(
                state.changed_files
            ),
            "active_staged_file": active,
            "active_staged_source": staged_source[-16000:],
            "last_tool_failure": state.last_tool_failure,
            "last_failure_signature": (
                state.last_failure_signature
            ),
            "verification_passed": (
                state.verification_passed
            ),
            "last_verification_command": (
                state.last_verification_command
            ),
            "recent_history": state.recent_context(
                limit=16
            ),
            "previous_recovery_guidance": (
                state.architect_recovery_guidance
            ),
        }

    def _run_architect_recovery(
        self,
        state: AgentState,
        reason: str,
    ) -> bool:
        if not self._architect_enabled():
            return False

        if (
            state.architect_recoveries
            >= self.config.architect_max_recoveries
        ):
            return False

        print(
            "[ARCHITECT] recovery escalation -> "
            "saving failure context"
        )

        package = self._architect_recovery_package(
            state,
            reason,
        )

        recovery_number = (
            state.architect_recoveries + 1
        )

        recovery_root = (
            self.project_root
            / ".helix"
            / "recovery"
        )
        recovery_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        context_path = recovery_root / (
            f"recovery-{recovery_number:03d}.json"
        )

        context_path.write_text(
            json.dumps(
                package,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        prompt = (
            "ARCHITECT RECOVERY CONSULTATION\n\n"
            "Helix's orchestrator and coder have reached a repeated failure "
            "or stall. They have been unloaded. You are now the senior "
            "architect/debugging consultant.\n\n"
            "Do NOT directly edit source code.\n"
            "Do NOT output Helix tool calls.\n"
            "Do NOT simply repeat previous attempted fixes.\n\n"
            "Produce an extremely detailed recovery guide containing:\n\n"
            "1. Most likely root cause(s), ranked by probability.\n"
            "2. Evidence supporting each diagnosis.\n"
            "3. Why the previous actions failed or stalled.\n"
            "4. Which exact file(s), functions, interfaces or state "
            "transitions are involved.\n"
            "5. A precise ordered recovery procedure.\n"
            "6. What the orchestrator should do next.\n"
            "7. What the coder should change or generate next.\n"
            "8. What NOT to do or repeat.\n"
            "9. Exact diagnostics/tests/commands that should be used to "
            "validate the repair.\n"
            "10. How to determine whether the root cause is model behavior, "
            "Helix orchestration, malformed source, incorrect tests, "
            "environment/infrastructure, or specification misunderstanding.\n"
            "11. Requirement regressions that must be checked afterward.\n"
            "12. Clear exit criteria for returning to normal implementation.\n"
            "13. Alternative recovery path if the primary fix fails.\n\n"
            "Use the original development blueprint as authoritative "
            "project intent, but correct any earlier assumption if the "
            "new diagnostic evidence proves it wrong.\n\n"
            "DEVELOPMENT BLUEPRINT:\n"
            + state.development_plan[-30000:]
            + "\n\nFAILURE PACKAGE:\n"
            + json.dumps(
                package,
                indent=2,
                ensure_ascii=False,
            )
        )

        guide = self._architect_request(prompt)

        guide_path = recovery_root / (
            f"recovery-{recovery_number:03d}.md"
        )

        guide_path.write_text(
            guide,
            encoding="utf-8",
            newline="\n",
        )

        state.architect_recovery_guidance = guide
        state.architect_recoveries = recovery_number
        state.last_architect_recovery_step = (
            state.step
        )

        # Give the normal pipeline a fresh opportunity using the guide.
        state.stall_steps = 0
        state.stall_level = 0

        print(
            "[ARCHITECT] recovery guide saved -> "
            f"{guide_path}"
        )
        print(
            "[ARCHITECT] returning control to "
            "orchestrator + coder"
        )

        return True

    def run(self, task: str) -> int:
        state = AgentState(task=task, project_root=self.project_root)
        self.router.reset_task_health()
        started = time.monotonic()
        self.console.task_start(task)

        try:
            self._create_development_plan(state)
        except Exception as exc:
            return self._stop_task(
                state,
                started,
                f"Architect planning failed: {exc}",
            )

        step = 0

        while True:
            if self.config.max_steps > 0 and step >= self.config.max_steps:
                state.step = step
                return self._stop_task(
                    state,
                    started,
                    "Maximum step count reached.",
                )

            step += 1
            state.step = step

            recovery_threshold = (
                self.config.architect_recovery_stall_steps
            )

            if (
                state.stall_steps >= recovery_threshold
                and (
                    step
                    - state.last_architect_recovery_step
                    >= recovery_threshold
                )
            ):
                last_reason = (
                    state.history[-1].summary
                    if state.history
                    else "Repeated Helix stall."
                )

                self._run_architect_recovery(
                    state,
                    last_reason,
                )

            preferred_role = self.router.choose_role(state)

            user_payload = json.dumps(
                {
                    "task": task,
                    "project_root": str(self.project_root),
                    "step": step,
                    "coding_failures": state.coding_failures,
                    "model_failures": state.model_failures,
                    "stall_steps": state.stall_steps,
                    "stall_level": state.stall_level,
                    "workspace_revision": state.workspace_revision,
                    "staged_files": [
                        {
                            "path": path,
                            "characters": characters,
                            "next_action": "append_file or finish_file",
                        }
                        for path, characters in sorted(state.staged_files.items())
                    ],
                    "staged_work_instruction": self._staged_work_instruction(state),
                    "active_staged_file": self._active_staged_file(state),
                    "staged_action_lock": (
                        {
                            "active": True,
                            "path": self._active_staged_file(state),
                            "characters": state.staged_files.get(
                                self._active_staged_file(state) or "",
                                0,
                            ),
                            "allowed_tools": (
                                ["append_file_raw", "append_file"]
                                if state.staged_files.get(
                                    self._active_staged_file(state) or "",
                                    0,
                                ) == 0
                                else [
                                    "append_file_raw",
                                    "append_file",
                                    "read_file",
                                    "finish_file",
                                ]
                            ),
                            "instruction": (
                                "You MUST continue the active staged file. "
                                "Do not scan, patch another file, run commands, "
                                "or begin another file."
                            ),
                        }
                        if self._active_staged_file(state)
                        else {"active": False}
                    ),
                    "last_completed_file": state.last_completed_file,
                    "post_finish_pending": state.post_finish_pending,
                    "architect_contract": state.architect_contract,
                    "architect_recovery_guidance": (
                        state.architect_recovery_guidance[-12000:]
                    ),
                    "architect_recoveries": state.architect_recoveries,
                    "remaining_requirements_instruction": self._remaining_requirements_instruction(state),
                    "post_finish_instruction": self._post_finish_instruction(state),
                    "last_tool_failure": state.last_tool_failure,
                    "failure_recovery_instruction": self._failure_recovery_instruction(state),
                    "verification_passed": state.verification_passed,
                    "recent_history": state.recent_context(),
                    "available_tools": self._available_tools_for_state(state),
                    "append_file_hard_limit": (
                        self.config.models.get("main").raw_max_chars
                        if self.config.models.get("main")
                        else 12000
                    ),
                    "append_file_recommended_size": (
                        self.config.models.get("main").raw_target_chars
                        if self.config.models.get("main")
                        else 7000
                    ),
                },
                indent=2,
            )

            try:
                staged_repair = 0
                accumulated_model_failures = 0
                request_payload = user_payload

                while True:
                    used_role, action = self.router.action_with_fallback(
                        preferred_role,
                        self.system_prompt,
                        request_payload,
                        coder_system=self.coder_system_prompt,
                    )
                    accumulated_model_failures += (
                        self.router.last_model_failures
                    )

                    tool = str(action["tool"])
                    args = action["args"]

                    lock_kind = "staged"
                    lock_error = self._staged_action_lock_error(
                        tool,
                        args,
                        state,
                    )

                    if not lock_error:
                        lock_kind = "post-finish"
                        lock_error = self._post_finish_action_lock_error(
                            tool,
                            args,
                            state,
                        )

                    if not lock_error:
                        break

                    # Give the model one correction attempt. After that,
                    # recover deterministically when Helix already knows the
                    # only valid continuation.
                    if staged_repair >= 1:
                        active_staged = self._active_staged_file(state)

                        if active_staged:
                            staged_characters = state.staged_files.get(
                                active_staged,
                                0,
                            )

                            if staged_characters == 0:
                                print(
                                    "[CORE] empty staged recovery -> "
                                    f"append_file_raw {active_staged}"
                                )

                                recovery_role = (
                                    "coder"
                                    if self.router._new_role_architecture()
                                    else "main"
                                )
                                backend = self.router.backend(
                                    recovery_role
                                )

                                raw_content = (
                                    self.router._complete_raw_content(
                                        backend,
                                        role=recovery_role,
                                        system=(
                                            self.coder_system_prompt
                                            if recovery_role == "coder"
                                            else self.system_prompt
                                        ),
                                        original_user=user_payload,
                                        path=active_staged,
                                    )
                                )

                                tool = "append_file_raw"
                                args = {
                                    "path": active_staged,
                                    "_raw_content": raw_content,
                                }
                                action = {
                                    "tool": tool,
                                    "args": args,
                                }

                                validate_action(action)
                                used_role = "main"
                                break

                            validation_failure = (
                                isinstance(
                                    state.last_tool_failure,
                                    dict,
                                )
                                and state.last_tool_failure.get("tool")
                                == "finish_file"
                                and state.last_tool_failure.get("path")
                                == active_staged
                                and state.last_tool_failure.get("stage")
                                == "staged_validation"
                            )

                            if not validation_failure:
                                print(
                                    f"[CORE] staged recovery -> finish_file "
                                    f"{active_staged}"
                                )
                                tool = "finish_file"
                                args = {"path": active_staged}
                                action = {
                                    "tool": tool,
                                    "args": args,
                                }
                                break

                        if (
                            lock_kind == "post-finish"
                            and self._post_finish_requires_test_file(
                                state
                            )
                        ):
                            test_path = self._post_finish_test_path(
                                state
                            )

                            print(
                                "[CORE] post-finish recovery -> "
                                f"begin_file {test_path}"
                            )

                            tool = "begin_file"
                            args = {"path": test_path}
                            action = {
                                "tool": tool,
                                "args": args,
                            }
                            break

                        if lock_kind == "post-finish":
                            verification_command = (
                                self._post_finish_verification_command(
                                    state
                                )
                            )

                            if verification_command:
                                print(
                                    "[CORE] post-finish recovery -> "
                                    f"run_command {verification_command}"
                                )

                                tool = "run_command"
                                args = {
                                    "command": verification_command,
                                    "verification": True,
                                }
                                action = {
                                    "tool": tool,
                                    "args": args,
                                }
                                used_role = "main"
                                break

                        raise HelixError(
                            "Model repeatedly violated the "
                            f"{lock_kind} action lock. "
                            + lock_error
                        )

                    staged_repair += 1

                    repair_name = (
                        "staged-action"
                        if lock_kind == "staged"
                        else "post-finish-action"
                    )

                    print(
                        f"[MODEL] {used_role} {repair_name} repair "
                        f"{staged_repair}/1: "
                        f"{lock_error}"
                    )

                    payload = json.loads(user_payload)
                    payload["action_lock_repair"] = {
                        "attempt": staged_repair,
                        "kind": lock_kind,
                        "error": lock_error,
                        "instruction": (
                            "Correct ONLY the next action. "
                            "Follow the current action lock exactly. "
                            "Do not scan, inspect unrelated files, or repeat "
                            "the rejected action."
                        ),
                    }

                    request_payload = json.dumps(
                        payload,
                        indent=2,
                    )

                state.model_failures += accumulated_model_failures
                signature = self._action_signature(tool, args)

                self.console.step(step, used_role, tool)

                if (
                    self._duplicate_sensitive(tool)
                    and signature in state.seen_actions
                ):
                    summary = (
                        f"Duplicate no-progress action blocked: {tool}. "
                        "Repository state has not changed; choose a different action."
                    )
                    state.add(used_role, tool, False, summary)
                    self.console.result(False, summary)

                    if self._mark_stall(
                        state,
                        f"Repeated {tool} without repository progress.",
                    ):
                        return self._stop_task(
                            state,
                            started,
                            "Helix stopped after repeated no-progress actions.",
                        )
                    continue

                ok, summary = self.execute(tool, args, state)
                state.add(used_role, tool, ok, summary)
                self.console.result(ok, summary)

                if self._duplicate_sensitive(tool) and ok:
                    state.seen_actions.add(signature)

                if tool == "finish" and ok:
                    duration = time.monotonic() - started
                    self.sessions.record(
                        task=task,
                        status="pass",
                        summary=summary,
                        steps=step,
                        duration_seconds=duration,
                    )
                    self.console.complete(summary, duration)
                    return 0

                if ok:
                    self._mark_progress(state)
                elif self._note_failure(state, tool, args, summary):
                    return self._stop_task(
                        state,
                        started,
                        "Helix stopped after the same failure repeated without progress.",
                    )

            except Exception as exc:
                failures = max(1, self.router.last_model_failures)
                state.model_failures += failures
                state.add(
                    preferred_role,
                    "agent_error",
                    False,
                    str(exc),
                )
                self.console.model_error(step, str(exc))

                if self._mark_stall(
                    state,
                    "Model/tool protocol failed without producing a usable action.",
                ):
                    return self._stop_task(
                        state,
                        started,
                        "Helix stopped after repeated model/protocol failures.",
                    )

    def execute(self, tool: str, args: dict[str, Any], state: AgentState) -> tuple[bool, str]:
        staged_guard = self._staged_work_guard(
            tool,
            args,
            state,
        )
        if staged_guard:
            return False, staged_guard

        post_finish_guard = self._post_finish_guard(
            tool,
            args,
            state,
        )
        if post_finish_guard:
            return False, post_finish_guard

        if tool == "list_files":
            if (
                state.last_tool_failure
                and state.last_tool_failure.get("tool") == "patch_file"
            ):
                path = state.last_tool_failure.get("path", "the failed file")
                return (
                    False,
                    f"Patch recovery required for {path}. "
                    "Do not rescan the repository. Read the failed file and "
                    "correct the patch_file action first.",
                )

            return True, json.dumps(self.workspace.list_files(), indent=2)

        if tool == "read_file":
            result = self.workspace.read_file(
                str(args["path"]),
                int(args.get("start_line", 1)),
                int(args["end_line"]) if args.get("end_line") is not None else None
            )
            return True, result

        if tool == "search_text":
            return True, json.dumps(
                self.workspace.search_text(str(args["query"])),
                indent=2
            )

        if tool == "hash_file":
            relative = str(args["path"])
            return True, self.workspace.hash_file(relative)

        if tool == "begin_file":
            if not self.permissions.write:
                return False, "Write denied: Helix is running in read-only mode."

            relative = self.workspace.normalize_relative(str(args["path"]))
            existing_characters = self.workspace.staged_file_characters(relative)

            if existing_characters is not None:
                state.staged_files[relative] = existing_characters
                return True, (
                    f"Already staged: {relative} "
                    f"({existing_characters} characters). "
                    "Continue with append_file or finish_file."
                )

            self.workspace.create_file_chunk(relative, "", "start")
            state.staged_files[relative] = 0
            state.workspace_revision += 1
            state.seen_actions.clear()
            state.post_finish_pending = False
            return True, f"Started staged file: {relative}"

        if tool == "append_file_raw":
            if not self.permissions.write:
                return False, "Write denied: Helix is running in read-only mode."

            relative = self.workspace.normalize_relative(str(args["path"]))
            content = args.get("_raw_content")

            if not isinstance(content, str):
                return (
                    False,
                    "append_file_raw requires literal source content between "
                    "<<<HELIX_CONTENT and HELIX_CONTENT.",
                )

            main_spec = self.config.models.get("main")
            raw_max_chars = (
                main_spec.raw_max_chars
                if main_spec is not None
                else 12000
            )

            if len(content) > raw_max_chars:
                return (
                    False,
                    f"Raw append payload too large: {len(content)} characters. "
                    f"Hard maximum is {raw_max_chars}. "
                    "Retry with a smaller coherent literal source response.",
                )

            overlap = self.workspace.staged_append_overlap(
                relative,
                content,
            )
            if overlap > 0:
                return (
                    False,
                    f"Duplicate staged content detected for {relative}: "
                    f"the proposed raw append overlaps {overlap} character(s) "
                    "already present in the staged file. "
                    "Continue only with genuinely new source code.",
                )

            self.workspace.create_file_chunk(
                relative,
                content,
                "append",
            )

            characters = self.workspace.staged_file_characters(relative)
            if characters is None:
                raise HelixError(
                    f"Staged file unexpectedly disappeared: {relative}"
                )

            state.staged_files[relative] = characters
            state.workspace_revision += 1
            state.seen_actions.clear()
            state.last_tool_failure = None

            return True, (
                f"Staged {len(content)} raw characters for {relative}; "
                f"{characters} total characters staged."
            )

        if tool == "append_file":
            if not self.permissions.write:
                return False, "Write denied: Helix is running in read-only mode."

            relative = self.workspace.normalize_relative(str(args["path"]))
            content = str(args.get("content", ""))

            if len(content) > self.config.max_create_chunk_chars:
                return (
                    False,
                    f"Append payload too large: {len(content)} characters. "
                    f"Hard maximum is {self.config.max_create_chunk_chars}. "
                    "Retry the same append_file action with at most 2500 characters.",
                )

            overlap = self.workspace.staged_append_overlap(
                relative,
                content,
            )
            if overlap > 0:
                return (
                    False,
                    f"Duplicate staged content detected for {relative}: "
                    f"the proposed append overlaps {overlap} character(s) "
                    "already present in the staged file. "
                    "Do not append the repeated content again. "
                    "Continue from the first genuinely new content; "
                    "use read_file if you need to inspect the staged file.",
                )

            self.workspace.create_file_chunk(
                relative,
                content,
                "append",
            )

            characters = self.workspace.staged_file_characters(relative)
            if characters is None:
                raise HelixError(
                    f"Staged file unexpectedly disappeared: {relative}"
                )

            state.staged_files[relative] = characters
            state.workspace_revision += 1
            state.seen_actions.clear()
            state.last_tool_failure = None

            return True, (
                f"Staged {len(content)} characters for {relative}; "
                f"{characters} total characters staged."
            )

        if tool == "finish_file":
            if not self.permissions.write:
                return False, "Write denied: Helix is running in read-only mode."

            relative = self.workspace.normalize_relative(str(args["path"]))
            assembled = self.workspace.create_file_chunk(
                relative,
                "",
                "finish",
            )

            if not isinstance(assembled, str):
                raise HelixError(
                    f"Unable to assemble staged file: {relative}"
                )

            validation_error = self._staged_python_validation_error(
                relative,
                assembled,
            )

            if validation_error:
                repaired, repair_error = self._repair_staged_python_file(
                    relative,
                    assembled,
                    validation_error,
                    state,
                )

                if repaired is None:
                    state.coding_failures += 1
                    state.last_tool_failure = {
                        "tool": "finish_file",
                        "path": relative,
                        "stage": "staged_validation",
                        "error": repair_error,
                    }

                    return False, (
                        repair_error
                        + " The file remains staged after exhausting "
                        "automatic full-file repair attempts."
                    )

                self.workspace.replace_staged_file(
                    relative,
                    repaired,
                )

                assembled = repaired
                state.staged_files[relative] = len(repaired)
                state.workspace_revision += 1
                state.seen_actions.clear()
                state.last_tool_failure = None

            result = self.forge.create_file(
                relative,
                assembled,
                state.step,
            )

            if (
                not result["ok"]
                and self.forge.infrastructure_failure(result)
            ):
                print(
                    f"[FORGE] Infrastructure/output encoding failure while "
                    f"creating {relative}; retrying once with UTF-8."
                )
                result = self.forge.create_file(
                    relative,
                    assembled,
                    state.step,
                )

            if result["ok"]:
                state.last_tool_failure = None
                self.workspace.clear_staged_file(relative)
                state.staged_files.pop(relative, None)
                state.last_completed_file = relative
                state.post_finish_pending = True
                state.changed_files.add(relative)
                state.workspace_revision += 1
                state.seen_actions.clear()
                state.verification_passed = False
                state.last_verification_command = ""
                digest = self.workspace.hash_file(relative)
                return True, (
                    f"Created via Patch Forge: {relative}; "
                    f"sha256={digest}"
                )

            state.coding_failures += 1
            return False, json.dumps(result, indent=2)

        if tool == "patch_file":
            if not self.permissions.write:
                return False, "Write denied: Helix is running in read-only mode."
            relative = str(args["path"])
            self.workspace.resolve(relative)
            result = self.forge.apply_edit(
                relative,
                str(args["op"]),
                state.step,
                old=args.get("old"),
                new=args.get("new"),
                anchor=args.get("anchor"),
                content=args.get("content"),
                expected_matches=int(args.get("expected_matches", 1)),
            )
            if result["ok"]:
                state.last_tool_failure = None
                state.post_finish_pending = False
                state.changed_files.add(relative)
                state.workspace_revision += 1
                state.seen_actions.clear()
                state.verification_passed = False
                state.last_verification_command = ""
                return True, f"Patch applied successfully: {relative}"

            state.coding_failures += 1
            state.last_tool_failure = {
                "tool": "patch_file",
                "path": relative,
                "stage": result.get("stage", "unknown"),
                "error": result.get("error", ""),
                "stdout": str(result.get("stdout", ""))[-6000:],
                "stderr": str(result.get("stderr", ""))[-6000:],
                "check_output": str(result.get("check_output", ""))[-6000:],
                "duplicate_text": str(result.get("duplicate_text", ""))[:2000],
            }
            return False, json.dumps(result, indent=2)

        if tool == "run_command":
            if not self.permissions.execute:
                return False, "Command execution denied by Helix permissions."

            command = str(args["command"])
            verification = bool(args.get("verification"))

            state.post_finish_pending = False

            if (
                verification
                and state.verification_passed
                and state.last_verification_command == command
            ):
                return True, f"Verification already passed; skipped duplicate: {command}"

            result = self.runner.run(
                command,
                int(args["timeout_seconds"]) if args.get("timeout_seconds") else None
            )

            if result["ok"]:
                if verification:
                    state.verification_passed = True
                    state.last_verification_command = command
                    state.coding_failures = 0

                    stdout = str(result.get("stdout", "")).strip()
                    last_line = stdout.splitlines()[-1] if stdout else ""
                    suffix = f" | {last_line}" if last_line else ""

                    return True, f"{command} -> exit 0{suffix}"

                return True, json.dumps(result, indent=2)

            state.coding_failures += 1

            if verification:
                state.verification_passed = False
                state.last_verification_command = ""

            return False, json.dumps(result, indent=2)

        if tool == "git_diff":
            result = self.runner.run("git --no-pager diff -- .")
            return bool(result["ok"]), json.dumps(result, indent=2)

        if tool == "finish":
            if state.changed_files and not state.verification_passed:
                return False, "Finish rejected: code changed but no successful verification has been recorded."
            summary = str(args.get("summary", "Task complete."))
            return True, summary

        raise HelixError(f"Unhandled tool: {tool}")


def load_benchmark_spec(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise HelixError("Benchmark file must contain a JSON object.")

    task = data.get("task")
    if not isinstance(task, str) or not task.strip():
        raise HelixError('Benchmark file requires a non-empty "task".')

    setup_files = data.get("setup_files", {})
    if not isinstance(setup_files, dict):
        raise HelixError('"setup_files" must be a JSON object.')

    return data


class _BenchmarkTee:
    def __init__(self, capture, terminal):
        self.capture = capture
        self.terminal = terminal

    def write(self, text):
        self.capture.write(text)
        self.terminal.write(text)
        self.terminal.flush()
        return len(text)

    def flush(self):
        self.capture.flush()
        self.terminal.flush()

    def isatty(self):
        return bool(
            getattr(self.terminal, "isatty", lambda: False)()
        )


def run_model_benchmark(args: argparse.Namespace) -> int:
    global _INTERRUPTIBLE_WAIT_HOOK

    spec_path = Path(args.benchmark).resolve()
    spec = load_benchmark_spec(spec_path)

    runs = max(1, int(args.runs))
    model_name = str(args.benchmark_model or "").strip()

    if not model_name:
        raise HelixError(
            "--benchmark-model is required when --benchmark is used."
        )

    base_project = Path(args.project).resolve()
    results_dir = (
        Path(args.benchmark_output).resolve()
        if args.benchmark_output
        else base_project / ".helix" / "benchmarks"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark_name = str(
        spec.get("name")
        or spec_path.stem
    )

    safe_model = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "-",
        model_name,
    )

    rows: list[dict[str, Any]] = []
    interrupted = False

    print()
    print("  HELIX MODEL BENCHMARK")
    print("  " + "-" * 64)
    print(f"  Benchmark {benchmark_name}")
    print(f"  Model     {model_name}")
    print(f"  Runs      {runs}")
    print(
        "  Fallback  "
        + (
            "enabled"
            if args.benchmark_allow_fallbacks
            else "disabled"
        )
    )
    print(
        "  Progress  "
        + (
            "enabled"
            if args.benchmark_progress
            else "quiet"
        )
    )
    print("  " + "-" * 64)

    for run_number in range(1, runs + 1):
        with tempfile.TemporaryDirectory(
            prefix="helix-benchmark-"
        ) as tmp:
            project = Path(tmp)

            for relative, content in spec.get(
                "setup_files",
                {},
            ).items():
                target = project / str(relative)
                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                target.write_text(
                    str(content),
                    encoding="utf-8",
                    newline="\n",
                )

            config = HelixConfig.load(
                Path(args.config)
            )

            existing = config.models.get("coder")
            config.models["coder"] = ModelSpec(
                provider=str(args.benchmark_provider),
                model=model_name,
                base_url=(
                    existing.base_url
                    if existing
                    else ""
                ),
                api_key_env=(
                    existing.api_key_env
                    if existing
                    else ""
                ),
                timeout_seconds=(
                    existing.timeout_seconds
                    if existing
                    else 180
                ),
                enabled=True,
                raw_target_chars=(
                    existing.raw_target_chars
                    if existing
                    else 7000
                ),
                raw_max_chars=(
                    existing.raw_max_chars
                    if existing
                    else 12000
                ),
                num_predict=(
                    existing.num_predict
                    if existing
                    else 8192
                ),
                num_ctx=(
                    existing.num_ctx
                    if existing
                    else 16384
                ),
            )

            if args.benchmark_orchestrator_model:
                old_orchestrator = config.models.get(
                    "orchestrator"
                )
                config.models["orchestrator"] = ModelSpec(
                    provider=str(args.benchmark_provider),
                    model=str(
                        args.benchmark_orchestrator_model
                    ),
                    base_url=(
                        old_orchestrator.base_url
                        if old_orchestrator
                        else ""
                    ),
                    timeout_seconds=(
                        old_orchestrator.timeout_seconds
                        if old_orchestrator
                        else 180
                    ),
                    enabled=True,
                    num_predict=(
                        old_orchestrator.num_predict
                        if old_orchestrator
                        else 2048
                    ),
                    num_ctx=(
                        old_orchestrator.num_ctx
                        if old_orchestrator
                        else 8192
                    ),
                )

            if args.benchmark_architect_model:
                old_architect = config.models.get(
                    "architect"
                )
                config.models["architect"] = ModelSpec(
                    provider=str(args.benchmark_provider),
                    model=str(
                        args.benchmark_architect_model
                    ),
                    base_url=(
                        old_architect.base_url
                        if old_architect
                        else ""
                    ),
                    timeout_seconds=(
                        old_architect.timeout_seconds
                        if old_architect
                        else 300
                    ),
                    enabled=True,
                    num_predict=(
                        old_architect.num_predict
                        if old_architect
                        else 12000
                    ),
                    num_ctx=(
                        old_architect.num_ctx
                        if old_architect
                        else 32768
                    ),
                )

            if not args.benchmark_allow_fallbacks:
                for role in ("remote",):
                    if role in config.models:
                        config.models[role].enabled = False

            prompt_path = (
                Path(__file__).parent
                / "prompts"
                / "orchestrator.txt"
            )

            agent = HelixAgent(
                project,
                config,
                prompt_path,
                AgentPermissions(),
            )

            capture = io.StringIO()
            started = time.monotonic()

            if args.benchmark_progress:
                print()
                print(
                    f"  ===== RUN {run_number}/{runs} "
                    f"- {model_name} ====="
                )

            original_hook = _INTERRUPTIBLE_WAIT_HOOK

            if args.benchmark_progress:
                def wait_hook(wait_seconds):
                    total = time.monotonic() - started
                    print(
                        f"  [BENCH] Run {run_number}/{runs} | "
                        f"{total:6.1f}s elapsed | "
                        f"waiting for model "
                        f"({wait_seconds:.1f}s)..."
                    )

                _INTERRUPTIBLE_WAIT_HOOK = wait_hook
            else:
                _INTERRUPTIBLE_WAIT_HOOK = None

            run_interrupted = False
            exit_code = None

            try:
                if args.benchmark_progress:
                    stream = _BenchmarkTee(
                        capture,
                        sys.stdout,
                    )
                    with contextlib.redirect_stdout(stream):
                        exit_code = agent.run(
                            str(spec["task"])
                        )
                else:
                    with contextlib.redirect_stdout(capture):
                        exit_code = agent.run(
                            str(spec["task"])
                        )

            except KeyboardInterrupt:
                run_interrupted = True
                interrupted = True

            finally:
                _INTERRUPTIBLE_WAIT_HOOK = original_hook

            duration = time.monotonic() - started
            output = capture.getvalue()

            verify_command = str(
                spec.get(
                    "verification_command",
                    "",
                )
            ).strip()

            verification_ok = None

            if (
                not run_interrupted
                and verify_command
            ):
                if args.benchmark_progress:
                    print(
                        f"  [BENCH] Final verification: "
                        f"{verify_command}"
                    )

                verify = CommandRunner(
                    project,
                    config.command_timeout_seconds,
                ).run(verify_command)

                verification_ok = bool(verify["ok"])

                if args.benchmark_progress:
                    print(
                        "  [BENCH] Verification "
                        + (
                            "PASS"
                            if verification_ok
                            else "FAIL"
                        )
                    )

            passed = (
                not run_interrupted
                and exit_code == 0
                and verification_ok is not False
            )

            status = (
                "interrupted"
                if run_interrupted
                else "pass"
                if passed
                else "fail"
            )

            step_matches = re.findall(
                r"\[(\d{3})\]",
                output,
            )
            steps = (
                max(int(value) for value in step_matches)
                if step_matches
                else 0
            )

            row = {
                "run": run_number,
                "status": status,
                "passed": passed,
                "agent_exit_code": exit_code,
                "verification_passed": verification_ok,
                "duration_seconds": round(duration, 2),
                "steps": steps,
                "raw_retries": output.count(
                    "raw-generation retry"
                ),
                "model_repairs": output.count(
                    "self-repair"
                ),
                "staged_repairs": output.count(
                    "staged-action repair"
                ),
                "stalls": output.count(
                    "LOOP   STALL"
                ),
                "escalations": output.count(
                    "ROUTER ESCALATE"
                ),
                "model_errors": output.count(
                    "MODEL  ERROR"
                ),
            }

            rows.append(row)

            log_path = (
                results_dir
                / (
                    f"{benchmark_name}-"
                    f"{safe_model}-run-"
                    f"{run_number}.log"
                )
            )

            log_path.write_text(
                output,
                encoding="utf-8",
                newline="\n",
            )

            print()
            print(
                f"  Run {run_number:02d}  "
                f"{status.upper():11} "
                f"{duration:.2f}s  "
                f"steps={steps}  "
                f"repairs={row['model_repairs']}  "
                f"raw={row['raw_retries']}  "
                f"staged={row['staged_repairs']}  "
                f"stalls={row['stalls']}"
            )

            if run_interrupted:
                print()
                print("  Benchmark interrupted by user.")
                print(
                    f"  Completed runs: "
                    f"{run_number - 1}/{runs}"
                )
                print(
                    f"  Interrupted run: "
                    f"{run_number}/{runs}"
                )
                print("  Partial run log saved.")
                break

    passed_count = sum(
        1 for row in rows if row["passed"]
    )

    completed_rows = [
        row
        for row in rows
        if row["status"] != "interrupted"
    ]

    average_duration = (
        sum(
            float(row["duration_seconds"])
            for row in completed_rows
        )
        / len(completed_rows)
        if completed_rows
        else 0.0
    )

    average_steps = (
        sum(
            int(row["steps"])
            for row in completed_rows
        )
        / len(completed_rows)
        if completed_rows
        else 0.0
    )

    summary = {
        "benchmark": benchmark_name,
        "model": model_name,
        "provider": args.benchmark_provider,
        "requested_runs": runs,
        "recorded_runs": len(rows),
        "completed_runs": len(completed_rows),
        "interrupted": interrupted,
        "passed": passed_count,
        "success_rate": (
            passed_count / len(completed_rows)
            if completed_rows
            else 0.0
        ),
        "average_duration_seconds": round(
            average_duration,
            2,
        ),
        "average_steps": round(
            average_steps,
            2,
        ),
        "total_raw_retries": sum(
            int(row["raw_retries"])
            for row in rows
        ),
        "total_model_repairs": sum(
            int(row["model_repairs"])
            for row in rows
        ),
        "total_staged_repairs": sum(
            int(row["staged_repairs"])
            for row in rows
        ),
        "total_stalls": sum(
            int(row["stalls"])
            for row in rows
        ),
        "total_escalations": sum(
            int(row["escalations"])
            for row in rows
        ),
        "results": rows,
    }

    result_base = (
        results_dir
        / f"{benchmark_name}-{safe_model}"
    )

    result_base.with_suffix(".json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    if rows:
        with result_base.with_suffix(".csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    print("  " + "-" * 64)
    print("  BENCHMARK SUMMARY")
    print("  " + "-" * 64)
    print(f"  Model       {model_name}")
    print(
        f"  Completed   "
        f"{len(completed_rows)}/{runs}"
    )
    print(
        f"  Passed      "
        f"{passed_count}/{len(completed_rows)}"
        if completed_rows
        else "  Passed      0/0"
    )
    print(
        f"  Success     "
        f"{summary['success_rate']:.1%}"
    )
    print(
        f"  Avg time    "
        f"{average_duration:.2f}s"
    )
    print(
        f"  Avg steps   "
        f"{average_steps:.1f}"
    )
    print(
        f"  Raw retries "
        f"{summary['total_raw_retries']}"
    )
    print(
        f"  Repairs     "
        f"{summary['total_model_repairs']}"
    )
    print(
        f"  Staged fix  "
        f"{summary['total_staged_repairs']}"
    )
    print(
        f"  Stalls      "
        f"{summary['total_stalls']}"
    )
    print(
        f"  Results     "
        f"{result_base.with_suffix('.json')}"
    )
    print()

    if interrupted:
        return 130

    return 0 if passed_count == runs else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helix autonomous coding agent")
    parser.add_argument("--project", default=".", help="Target project directory")
    parser.add_argument("--task", help="Run one task and exit; omit for interactive CLI")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("helix.toml")),
        help="Path to helix.toml"
    )
    parser.add_argument("--read-only", action="store_true", help="Block file creation and source edits")
    parser.add_argument("--no-exec", action="store_true", help="Block model-requested command execution")
    parser.add_argument(
        "--benchmark",
        help="Run a benchmark definition JSON instead of the interactive shell",
    )
    parser.add_argument(
        "--benchmark-model",
        help="Model to use as the dedicated CODER for benchmark runs",
    )
    parser.add_argument(
        "--benchmark-provider",
        default="ollama",
        help="Provider used by --benchmark-model (default: ollama)",
    )
    parser.add_argument(
        "--benchmark-orchestrator-model",
        help="Optional orchestrator model override for benchmarks",
    )
    parser.add_argument(
        "--benchmark-architect-model",
        help="Optional architect model override for benchmarks",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of isolated benchmark runs",
    )
    parser.add_argument(
        "--benchmark-output",
        help="Directory for benchmark JSON, CSV, and run logs",
    )
    parser.add_argument(
        "--benchmark-allow-fallbacks",
        action="store_true",
        help="Allow LARGE/REMOTE fallback during model benchmarks",
    )
    parser.add_argument(
        "--benchmark-progress",
        action="store_true",
        help="Show live Helix steps and model-wait heartbeat during benchmarks",
    )
    return parser


def collect_multiline_task(input_fn=None) -> str:
    if input_fn is None:
        input_fn = input

    print("  Multiline task mode. Paste/type instructions below.")
    print("  Enter :end on its own line to submit.")
    lines: list[str] = []

    while True:
        line = input_fn("... ")
        if line.strip().lower() == ":end":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def interactive_shell(agent: HelixAgent) -> int:
    print("  Commands: :help  :task  :models  :history  :clear  :quit")
    print()

    while True:
        try:
            task = input("helix> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n  Use :quit to exit Helix.")
            continue

        if not task:
            continue

        command = task.lower()

        if command in {":quit", ":exit"}:
            return 0

        if command == ":help":
            print("  Enter a one-line coding task directly.")
            print("  :task     Enter/paste a multiline task; finish with :end")
            print("  :models   Show configured models")
            print("  :history  Show recent tasks")
            print("  :clear    Clear the terminal")
            print("  :quit     Exit Helix")
            continue

        if command == ":task":
            try:
                task = collect_multiline_task()
            except (EOFError, KeyboardInterrupt):
                print("\n  Multiline task cancelled.")
                continue

            if not task:
                print("  Empty task cancelled.")
                continue

        elif command == ":models":
            agent.console.show_models()
            continue

        elif command == ":history":
            agent.console.show_history(agent.sessions.recent())
            continue

        elif command == ":clear":
            os.system("cls" if os.name == "nt" else "clear")
            agent.console.banner()
            continue

        elif task.startswith(":"):
            print(f"  Unknown command: {task}")
            continue

        try:
            agent.run(task)
        except KeyboardInterrupt:
            print("\n  Task interrupted by user.")


def main() -> int:
    args = build_parser().parse_args()
    project = Path(args.project)
    if not project.exists() or not project.is_dir():
        print(f"Project directory does not exist: {project}", file=sys.stderr)
        return 2
    config = HelixConfig.load(Path(args.config))
    prompt_path = Path(__file__).parent / "prompts" / "orchestrator.txt"
    if args.benchmark:
        return run_model_benchmark(args)
    permissions = AgentPermissions(
        write=not args.read_only,
        execute=not args.no_exec,
    )
    agent = HelixAgent(project, config, prompt_path, permissions)
    agent.console.banner()
    if args.task:
        return agent.run(args.task)
    return interactive_shell(agent)


if __name__ == "__main__":
    raise SystemExit(main())
