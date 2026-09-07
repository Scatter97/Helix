import tempfile
import json
import io
import unittest
from pathlib import Path

import helix


class HelixTests(unittest.TestCase):
    def test_extract_action_accepts_json(self):
        action = helix.extract_action('{"tool":"list_files","args":{}}')
        self.assertEqual(action["tool"], "list_files")

    def test_workspace_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = helix.Workspace(Path(tmp))
            with self.assertRaises(helix.HelixError):
                workspace.resolve("../outside.txt")

    def test_staging_refuses_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("existing\n", encoding="utf-8")
            workspace = helix.Workspace(root)

            with self.assertRaises(helix.HelixError):
                workspace.create_file_chunk("app.py", "", "start")

    def test_router_escalates_after_failures(self):
        config = helix.HelixConfig(
            patchforge_cli="",
            max_steps=10,
            command_timeout_seconds=10,
            models={
                "small": helix.ModelSpec("ollama", "small"),
                "main": helix.ModelSpec("ollama", "main"),
                "large": helix.ModelSpec("ollama", "large"),
                "remote": helix.ModelSpec("openai_compatible", "remote", enabled=True)
            }
        )
        router = helix.ModelRouter(config)
        state = helix.AgentState(task="x", project_root=Path("."))
        state.coding_failures = 3
        self.assertEqual(router.choose_role(state), "large")
        state.coding_failures = 5
        self.assertEqual(router.choose_role(state), "remote")

    def test_patchforge_apply_edit_builds_valid_python_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "calculator.py"
            source.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

            adapter = helix.PatchForgeAdapter(root, "")
            captured = {}

            def fake_apply(patch, attempt):
                captured["patch"] = patch
                captured["attempt"] = attempt
                return {"ok": True}

            adapter.apply = fake_apply

            result = adapter.apply_edit(
                "calculator.py",
                "replace",
                4,
                old="    return a - b\n",
                new="    return a + b\n",
                expected_matches=1,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(captured["patch"]["version"], 2)
            self.assertIs(
                captured["patch"]["options"]["rollback_on_verify_failure"],
                True,
            )
            self.assertEqual(
                captured["patch"]["files"][0]["edits"][0]["op"],
                "replace",
            )


    def test_workspace_ignores_pytest_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("print('ok')\n", encoding="utf-8")
            cache = root / ".pytest_cache"
            cache.mkdir()
            (cache / "ignored.txt").write_text("ignore\n", encoding="utf-8")
            files = helix.Workspace(root).list_files()
            self.assertIn("app.py", files)
            self.assertNotIn(".pytest_cache/ignored.txt", files)

    def test_duplicate_verification_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")
            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=10,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )
            agent = helix.HelixAgent(root, config, prompt)
            calls = []
            def fake_run(command, timeout_seconds=None):
                calls.append(command)
                return {
                    "ok": True,
                    "exit_code": 0,
                    "stdout": "1 passed\n",
                    "stderr": "",
                    "duration_seconds": 0.01,
                    "timed_out": False,
                }
            agent.runner.run = fake_run
            state = helix.AgentState(task="test", project_root=root)
            first_ok, _ = agent.execute("run_command", {"command": "pytest -q", "verification": True}, state)
            second_ok, summary = agent.execute("run_command", {"command": "pytest -q", "verification": True}, state)
            self.assertTrue(first_ok)
            self.assertTrue(second_ok)
            self.assertEqual(calls, ["pytest -q"])
            self.assertIn("skipped duplicate", summary)

    def test_read_only_blocks_create_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")
            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=10,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )
            agent = helix.HelixAgent(
                root,
                config,
                prompt,
                helix.AgentPermissions(write=False, execute=True),
            )
            state = helix.AgentState(task="test", project_root=root)
            ok, summary = agent.execute(
            "begin_file",
            {"path": "blocked.py"},
            state,
        )
            self.assertFalse(ok)
            self.assertIn("read-only", summary.lower())
            self.assertFalse((root / "blocked.py").exists())

    def test_session_history_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = helix.SessionStore(Path(tmp))
            store.record(
                task="Fix tests",
                status="pass",
                summary="Passed",
                steps=5,
                duration_seconds=1.25,
            )
            history = store.recent()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["task"], "Fix tests")
            self.assertEqual(history[0]["status"], "pass")



    def test_workspace_ignores_patchforge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("print('ok')\n", encoding="utf-8")
            checkpoint = root / ".patchforge" / "checkpoints" / "abc"
            checkpoint.mkdir(parents=True)
            (checkpoint / "old.py").write_text("old\n", encoding="utf-8")

            files = helix.Workspace(root).list_files()

            self.assertIn("app.py", files)
            self.assertFalse(
                any(path.startswith(".patchforge/") for path in files)
            )

    def test_chunked_file_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = helix.Workspace(root)

            self.assertIsNone(
                workspace.create_file_chunk("app.py", "def main():\n", "start")
            )
            self.assertIsNone(
                workspace.create_file_chunk("app.py", "    print(", "append")
            )
            assembled = workspace.create_file_chunk(
                "app.py",
                "'hello')\n",
                "finish",
            )

            self.assertEqual(
                assembled,
                "def main():\n    print('hello')\n",
            )
            self.assertFalse((root / "app.py").exists())

    def test_router_escalates_when_stalled(self):
        config = helix.HelixConfig(
            patchforge_cli="",
            max_steps=0,
            command_timeout_seconds=10,
            models={
                "main": helix.ModelSpec("ollama", "main"),
                "large": helix.ModelSpec("ollama", "large"),
                "remote": helix.ModelSpec(
                    "openai_compatible",
                    "remote",
                    enabled=True,
                ),
            },
        )
        router = helix.ModelRouter(config)
        state = helix.AgentState(task="x", project_root=Path("."))

        state.stall_level = 1
        self.assertEqual(router.choose_role(state), "large")

        state.stall_level = 2
        self.assertEqual(router.choose_role(state), "remote")

    def test_multiline_task_collection(self):
        lines = iter(
            [
                "Build an app.",
                "Requirements:",
                "- test it",
                ":end",
            ]
        )

        result = helix.collect_multiline_task(
            lambda prompt: next(lines)
        )

        self.assertEqual(
            result,
            "Build an app.\nRequirements:\n- test it",
        )


    def test_normalize_relative_accepts_absolute_in_project_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = helix.Workspace(root)
            absolute = root / "src" / "app.py"

            self.assertEqual(
                workspace.normalize_relative(str(absolute)),
                "src/app.py",
            )

    def test_normalize_relative_rejects_absolute_outside_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            workspace = helix.Workspace(root)
            outside = Path(tmp) / "outside.py"

            with self.assertRaises(helix.HelixError):
                workspace.normalize_relative(str(outside))

    def test_chunked_creation_accepts_absolute_project_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = helix.Workspace(root)
            target = root / "src" / "app.py"

            workspace.create_file_chunk(
                str(target),
                "print(",
                "start",
            )
            assembled = workspace.create_file_chunk(
                str(target),
                "'ok')\n",
                "finish",
            )

            self.assertEqual(assembled, "print('ok')\n")
            self.assertFalse(target.exists())

    def test_oversized_chunk_reports_actual_size_and_retry_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")
            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
                max_create_chunk_chars=10,
            )
            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)

            begin_ok, _ = agent.execute(
                "begin_file",
                {"path": "app.py"},
                state,
            )
            self.assertTrue(begin_ok)

            ok, summary = agent.execute(
                "append_file",
                {
                    "path": "app.py",
                    "content": "12345678901",
                },
                state,
            )

            self.assertFalse(ok)
            self.assertIn("11 characters", summary)
            self.assertIn("2500", summary)

    def test_patchforge_create_file_builds_create_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = helix.PatchForgeAdapter(root, "")
            captured = {}

            def fake_apply(patch, attempt):
                captured["patch"] = patch
                captured["attempt"] = attempt
                return {"ok": True}

            adapter.apply = fake_apply

            result = adapter.create_file(
                "src/new_file.py",
                "print('hello')\n",
                7,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(captured["attempt"], 7)
            self.assertEqual(captured["patch"]["version"], 2)

            entry = captured["patch"]["files"][0]
            self.assertEqual(entry["path"], "src/new_file.py")
            self.assertEqual(entry["action"], "create")
            self.assertEqual(entry["content"], "print('hello')\n")
            self.assertNotIn("sha256", entry)
            self.assertNotIn("edits", entry)

    def test_finish_file_routes_creation_through_patchforge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")
            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )
            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)
            captured = {}

            def fake_create(relative, content, attempt):
                captured["relative"] = relative
                captured["content"] = content
                captured["attempt"] = attempt
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_text(
                    content,
                    encoding="utf-8",
                    newline="\n",
                )
                return {"ok": True}

            agent.forge.create_file = fake_create

            begin_ok, _ = agent.execute(
                "begin_file",
                {"path": "src/app.py"},
                state,
            )
            append_ok, _ = agent.execute(
                "append_file",
                {
                    "path": "src/app.py",
                    "content": "print('ok')\n",
                },
                state,
            )
            finish_ok, summary = agent.execute(
                "finish_file",
                {"path": "src/app.py"},
                state,
            )

            self.assertTrue(begin_ok)
            self.assertTrue(append_ok)
            self.assertTrue(finish_ok)
            self.assertEqual(captured["relative"], "src/app.py")
            self.assertEqual(captured["content"], "print('ok')\n")
            self.assertIn("Created via Patch Forge", summary)
            self.assertTrue((root / "src" / "app.py").exists())
            self.assertNotIn("src/app.py", state.staged_files)

    def test_read_file_reads_staged_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = helix.Workspace(root)

            workspace.create_file_chunk(
                "new.py",
                "print('staged')\n",
                "start",
            )

            result = workspace.read_file("new.py")

            self.assertIn("print('staged')", result)
            self.assertFalse((root / "new.py").exists())

    def test_begin_file_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)

            first_ok, _ = agent.execute(
                "begin_file",
                {"path": "app.py"},
                state,
            )
            revision_after_first = state.workspace_revision

            second_ok, summary = agent.execute(
                "begin_file",
                {"path": "app.py"},
                state,
            )

            self.assertTrue(first_ok)
            self.assertTrue(second_ok)
            self.assertIn("Already staged", summary)
            self.assertEqual(state.staged_files["app.py"], 0)
            self.assertEqual(
                state.workspace_revision,
                revision_after_first,
            )

    def test_append_file_updates_staged_state_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)

            agent.execute(
                "begin_file",
                {"path": "app.py"},
                state,
            )

            revision_before_append = state.workspace_revision
            state.seen_actions.add("old-action")

            ok, summary = agent.execute(
                "append_file",
                {
                    "path": "app.py",
                    "content": "print('hello')\n",
                },
                state,
            )

            self.assertTrue(ok)
            self.assertEqual(
                state.staged_files["app.py"],
                len("print('hello')\n"),
            )
            self.assertGreater(
                state.workspace_revision,
                revision_before_append,
            )
            self.assertEqual(state.seen_actions, set())
            self.assertIn("total characters staged", summary)

    def test_staged_work_instruction_reports_unfinished_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)
            state.staged_files["storage.py"] = 1234

            guidance = agent._staged_work_instruction(state)

            self.assertIn("storage.py", guidance)
            self.assertIn("1234 characters", guidance)
            self.assertIn("append_file", guidance)
            self.assertIn("finish_file", guidance)
            self.assertIn("Do not call begin_file again", guidance)

    def test_extract_action_reports_empty_response(self):
        with self.assertRaises(helix.HelixError) as context:
            helix.extract_action("   ")

        self.assertIn("empty response", str(context.exception))

    def test_extract_action_reports_truncated_json_error(self):
        with self.assertRaises(helix.HelixError) as context:
            helix.extract_action(
                '{"tool":"append_file","args":{"path":"todo.py","content":"abc'
            )

        message = str(context.exception)
        self.assertIn("Invalid model JSON", message)

    def test_validate_action_reports_missing_required_argument(self):
        action = {
            "tool": "append_file",
            "args": {"path": "todo.py"},
        }

        with self.assertRaises(helix.HelixError) as context:
            helix.validate_action(action)

        self.assertIn('missing required argument "content"', str(context.exception))

    def test_validate_action_reports_unknown_tool(self):
        with self.assertRaises(helix.HelixError) as context:
            helix.validate_action(
                {"tool": "write_file", "args": {}}
            )

        self.assertIn("Unknown Helix tool", str(context.exception))
        self.assertIn("append_file", str(context.exception))

    def test_repair_prompt_preserves_staged_context(self):
        original_user = json.dumps(
            {
                "task": "Build app",
                "staged_files": [
                    {
                        "path": "todo.py",
                        "characters": 3385,
                        "next_action": "append_file or finish_file",
                    }
                ],
            }
        )

        prompt = helix.build_model_repair_prompt(
            original_user=original_user,
            raw_response="",
            error="Model returned an empty response.",
            attempt=1,
            max_attempts=2,
        )

        self.assertIn("todo.py", prompt)
        self.assertIn("3385", prompt)
        self.assertIn("EMPTY RESPONSE", prompt)
        self.assertIn("Do not rescan", prompt)
        self.assertIn("continue the currently staged work", prompt)

    def test_router_repairs_invalid_json_before_fallback(self):
        config = helix.HelixConfig(
            patchforge_cli="",
            max_steps=0,
            command_timeout_seconds=10,
            models={
                "main": helix.ModelSpec("ollama", "main"),
                "large": helix.ModelSpec("ollama", "large"),
            },
            model_repair_attempts=2,
        )
        router = helix.ModelRouter(config)

        class FakeBackend:
            def __init__(self):
                self.calls = []

            def complete(
                self,
                system,
                user,
                *,
                json_mode=True,
            ):
                self.calls.append(user)
                if len(self.calls) == 1:
                    return ""
                if len(self.calls) == 2:
                    return '{"tool":"append_file","args":{"path":"todo.py"}}'
                return json.dumps(
                    {
                        "tool": "append_file",
                        "args": {
                            "path": "todo.py",
                            "content": "new content",
                        },
                    }
                )

        main_backend = FakeBackend()
        large_called = {"value": False}

        class LargeBackend:
            def complete(
                self,
                system,
                user,
                *,
                json_mode=True,
            ):
                large_called["value"] = True
                return json.dumps(
                    {"tool": "list_files", "args": {}}
                )

        def fake_backend(role):
            if role == "main":
                return main_backend
            return LargeBackend()

        router.backend = fake_backend

        user = json.dumps(
            {
                "task": "Build todo",
                "staged_files": [
                    {"path": "todo.py", "characters": 1000}
                ],
            }
        )

        role, action = router.action_with_fallback(
            "main",
            "system",
            user,
        )

        self.assertEqual(role, "main")
        self.assertEqual(action["tool"], "append_file")
        self.assertEqual(len(main_backend.calls), 3)
        self.assertFalse(large_called["value"])
        self.assertIn("todo.py", main_backend.calls[1])
        self.assertIn("empty response", main_backend.calls[1].lower())
        self.assertIn("missing required argument", main_backend.calls[2])

    def test_duplicate_append_overlap_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = helix.Workspace(root)

            existing = (
                "def first():\n"
                "    return 1\n\n"
                + "x" * 120
            )
            workspace.create_file_chunk(
                "app.py",
                existing,
                "start",
            )

            duplicate = ("x" * 100) + "\ndef second():\n    return 2\n"

            overlap = workspace.staged_append_overlap(
                "app.py",
                duplicate,
            )

            self.assertGreaterEqual(overlap, 100)

    def test_append_file_rejects_duplicate_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )
            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)

            begin_ok, _ = agent.execute(
                "begin_file",
                {"path": "app.py"},
                state,
            )
            self.assertTrue(begin_ok)

            first = "A" * 150
            append_ok, _ = agent.execute(
                "append_file",
                {"path": "app.py", "content": first},
                state,
            )
            self.assertTrue(append_ok)

            revision_before = state.workspace_revision

            duplicate_ok, summary = agent.execute(
                "append_file",
                {
                    "path": "app.py",
                    "content": ("A" * 100) + "new",
                },
                state,
            )

            self.assertFalse(duplicate_ok)
            self.assertIn("Duplicate staged content detected", summary)
            self.assertEqual(
                state.workspace_revision,
                revision_before,
            )
            self.assertEqual(
                state.staged_files["app.py"],
                150,
            )

    def test_insert_after_duplicate_adjacent_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "app.py"
            source.write_text(
                "def add():\n"
                "    item = {\n"
                "        'created': True\n"
                "    }\n"
                "    save(item)\n"
                "    return item\n",
                encoding="utf-8",
            )

            adapter = helix.PatchForgeAdapter(root, "")

            result = adapter.apply_edit(
                "app.py",
                "insert_after",
                1,
                anchor="        'created': True\n",
                content=(
                    "    }\n"
                    "    save(item)\n"
                    "    return item\n\n"
                    "def next_function():\n"
                    "    pass\n"
                ),
                expected_matches=1,
            )

            self.assertFalse(result["ok"])
            self.assertEqual(result["stage"], "precheck")
            self.assertIn("duplicates source text", result["error"])
            self.assertIn("save(item)", result["duplicate_text"])

    def test_python_patch_adds_py_compile_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "app.py"
            source.write_text(
                "def value():\n    return 1\n",
                encoding="utf-8",
            )

            adapter = helix.PatchForgeAdapter(root, "")
            captured = {}

            def fake_apply(patch, attempt):
                captured["patch"] = patch
                return {"ok": True}

            adapter.apply = fake_apply

            result = adapter.apply_edit(
                "app.py",
                "replace",
                2,
                old="    return 1\n",
                new="    return 2\n",
                expected_matches=1,
            )

            self.assertTrue(result["ok"])
            commands = [
                item["command"]
                for item in captured["patch"]["verify"]
            ]
            self.assertIn("git diff --check", commands)
            self.assertTrue(
                any(
                    command.startswith("python -m py_compile")
                    and "app.py" in command
                    for command in commands
                )
            )

    def test_patch_failure_recovery_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)
            state.last_tool_failure = {
                "tool": "patch_file",
                "path": "todo.py",
                "stage": "apply",
                "stderr": "SyntaxError",
            }

            instruction = agent._failure_recovery_instruction(state)

            self.assertIn("PATCH RECOVERY REQUIRED", instruction)
            self.assertIn("todo.py", instruction)
            self.assertIn("Do not call list_files", instruction)
            self.assertIn("read_file", instruction)

    def test_list_files_blocked_during_patch_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")
            (root / "app.py").write_text("pass\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)
            state.last_tool_failure = {
                "tool": "patch_file",
                "path": "app.py",
                "stage": "apply",
            }

            ok, summary = agent.execute(
                "list_files",
                {},
                state,
            )

            self.assertFalse(ok)
            self.assertIn("Patch recovery required", summary)
            self.assertIn("Read the failed file", summary)

    def test_successful_patch_clears_failure_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")
            (root / "app.py").write_text(
                "value = 1\n",
                encoding="utf-8",
            )

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)
            state.last_tool_failure = {
                "tool": "patch_file",
                "path": "app.py",
                "stage": "apply",
            }

            agent.forge.apply_edit = lambda *args, **kwargs: {"ok": True}

            ok, _ = agent.execute(
                "patch_file",
                {
                    "path": "app.py",
                    "op": "replace",
                    "old": "value = 1\n",
                    "new": "value = 2\n",
                    "expected_matches": 1,
                },
                state,
            )

            self.assertTrue(ok)
            self.assertIsNone(state.last_tool_failure)

    def test_active_staged_file_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)
            state.staged_files["cli.py"] = 120

            self.assertEqual(
                agent._active_staged_file(state),
                "cli.py",
            )

    def test_list_files_blocked_while_file_is_staged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)
            state.staged_files["cli.py"] = 120

            ok, summary = agent.execute(
                "list_files",
                {},
                state,
            )

            self.assertFalse(ok)
            self.assertIn("cli.py", summary)
            self.assertIn("blocked", summary)

    def test_unrelated_read_blocked_while_file_is_staged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)
            state.staged_files["cli.py"] = 120

            ok, summary = agent.execute(
                "read_file",
                {"path": "calculator.py"},
                state,
            )

            self.assertFalse(ok)
            self.assertIn("Unrelated read blocked", summary)
            self.assertIn("cli.py", summary)
            self.assertIn("calculator.py", summary)

    def test_active_staged_file_can_still_be_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)

            agent.workspace.create_file_chunk(
                "cli.py",
                "print('hello')\n",
                "start",
            )
            state.staged_files["cli.py"] = len("print('hello')\n")

            ok, summary = agent.execute(
                "read_file",
                {"path": "cli.py"},
                state,
            )

            self.assertTrue(ok)
            self.assertIn("hello", summary)

    def test_second_begin_file_blocked_during_staged_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)
            state.staged_files["cli.py"] = 120

            ok, summary = agent.execute(
                "begin_file",
                {"path": "README.md"},
                state,
            )

            self.assertFalse(ok)
            self.assertIn("cli.py", summary)
            self.assertIn("README.md", summary)

    def test_tool_schema_for_repair_returns_specific_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)

            schema = agent._tool_schema_for_repair(
                {
                    "tool": "run_command",
                    "args": {
                        "cmd": "pytest -q",
                        "cwd": ".",
                    },
                }
            )

            self.assertIn('"run_command"', schema)
            self.assertIn('"command"', schema)
            self.assertNotIn('"cmd"', schema)
            self.assertNotIn('"cwd"', schema)

    def test_extract_action_accepts_raw_file_transport(self):
        response = (
            '{"tool":"append_file_raw","args":{"path":"app.py"}}\n'
            "<<<HELIX_CONTENT\n"
            'def hello():\n'
            '    print("hello")\n'
            '    text = """triple quotes work"""\n'
            "HELIX_CONTENT"
        )

        action = helix.extract_action(response)

        self.assertEqual(action["tool"], "append_file_raw")
        self.assertEqual(action["args"]["path"], "app.py")
        self.assertIn('print("hello")', action["args"]["_raw_content"])
        self.assertIn('"""triple quotes work"""', action["args"]["_raw_content"])

        helix.validate_action(action)

    def test_raw_file_transport_requires_closing_marker(self):
        response = (
            '{"tool":"append_file_raw","args":{"path":"app.py"}}\n'
            "<<<HELIX_CONTENT\n"
            "print('hello')\n"
        )

        with self.assertRaises(helix.HelixError) as context:
            helix.extract_action(response)

        self.assertIn("closing", str(context.exception))

    def test_raw_transport_rejects_wrong_tool(self):
        response = (
            '{"tool":"append_file","args":{"path":"app.py","content":""}}\n'
            "<<<HELIX_CONTENT\n"
            "print('hello')\n"
            "HELIX_CONTENT"
        )

        with self.assertRaises(helix.HelixError) as context:
            helix.extract_action(response)

        self.assertIn("append_file_raw", str(context.exception))

    def test_append_file_raw_updates_staged_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)

            ok, _ = agent.execute(
                "begin_file",
                {"path": "app.py"},
                state,
            )
            self.assertTrue(ok)

            content = (
                'def hello():\n'
                '    print("hello")\n'
            )

            ok, summary = agent.execute(
                "append_file_raw",
                {
                    "path": "app.py",
                    "_raw_content": content,
                },
                state,
            )

            self.assertTrue(ok)
            self.assertIn("raw characters", summary)

            staged = agent.workspace.read_file("app.py")
            self.assertIn('print("hello")', staged)

    def test_repair_guidance_recommends_raw_transport_for_append(self):
        broken = (
            '{"tool":"append_file","args":{"path":"app.py",'
            '"content":"print("hello")"}}'
        )

        guidance = helix.repair_schema_guidance(broken)

        self.assertIn("append_file_raw", guidance)
        self.assertIn('{"tool":"append_file_raw"', guidance)
        self.assertIn("Do not include source code in this JSON response", guidance)

    def test_escalation_handoff_preserves_shared_state(self):
        payload = json.dumps(
            {
                "task": "Build a todo app",
                "changed_files": ["todo.py"],
                "staged_files": [
                    {
                        "path": "tests.py",
                        "characters": 800,
                    }
                ],
                "active_staged_file": "tests.py",
                "last_completed_file": "todo.py",
                "last_tool_failure": None,
                "verification_passed": False,
                "recent_history": [
                    {
                        "tool": "finish_file",
                        "ok": True,
                        "summary": "Created todo.py",
                    }
                ],
            }
        )

        handoff = helix.build_escalation_handoff(
            original_user=payload,
            previous_errors=["main returned malformed JSON"],
        )

        self.assertIn("ESCALATION HANDOFF", handoff)
        self.assertIn("Build a todo app", handoff)
        self.assertIn("tests.py", handoff)
        self.assertIn("todo.py", handoff)
        self.assertIn("malformed JSON", handoff)
        self.assertIn("Continue the active staged file", handoff)
        self.assertIn("CURRENT SHARED HELIX STATE", handoff)

    def test_finish_file_records_last_completed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)

            agent.workspace.create_file_chunk(
                "app.py",
                "",
                "start",
            )
            agent.workspace.create_file_chunk(
                "app.py",
                "print('ok')\n",
                "append",
            )
            state.staged_files["app.py"] = len("print('ok')\n")

            agent.forge.create_file = (
                lambda relative, content, attempt: {"ok": True}
            )
            agent.workspace.hash_file = lambda relative: "deadbeef"

            ok, _ = agent.execute(
                "finish_file",
                {"path": "app.py"},
                state,
            )

            self.assertTrue(ok)
            self.assertEqual(
                state.last_completed_file,
                "app.py",
            )

    def test_post_finish_instruction_discourages_repeat_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)
            state = helix.AgentState(task="test", project_root=root)
            state.last_completed_file = "todo.py"

            guidance = agent._post_finish_instruction(state)

            self.assertIn("todo.py", guidance)
            self.assertIn("Treat it as completed work", guidance)
            self.assertIn("Do not call list_files", guidance)
            self.assertIn("next concrete implementation", guidance)

    def test_raw_content_prompt_requests_literal_source_only(self):
        payload = json.dumps(
            {
                "task": "Build app",
                "active_staged_file": "app.py",
                "staged_files": [
                    {
                        "path": "app.py",
                        "characters": 100,
                    }
                ],
                "recent_history": [],
            }
        )

        prompt = helix.build_raw_content_prompt(
            original_user=payload,
            path="app.py",
        )

        self.assertIn("app.py", prompt)
        self.assertIn("Return ONLY literal file content", prompt)
        self.assertIn("Do NOT return JSON", prompt)
        self.assertIn("Do NOT return markdown fences", prompt)

    def test_ollama_json_mode_is_optional(self):
        spec = helix.ModelSpec(
            provider="ollama",
            model="test-model",
        )
        backend = helix.OllamaBackend(spec)

        self.assertIsInstance(backend, helix.ModelBackend)

    def test_router_two_phase_raw_generation(self):
        class FakeBackend:
            def __init__(self):
                self.calls = []

            def complete(
                self,
                system,
                user,
                *,
                json_mode=True,
            ):
                self.calls.append(json_mode)

                if json_mode:
                    return json.dumps(
                        {
                            "tool": "append_file_raw",
                            "args": {
                                "path": "app.py",
                            },
                        }
                    )

                return (
                    'def hello():\n'
                    '    print("hello")\n'
                )

        config = helix.HelixConfig(
            patchforge_cli="",
            max_steps=0,
            command_timeout_seconds=10,
            models={
                "main": helix.ModelSpec(
                    provider="ollama",
                    model="main",
                )
            },
        )

        router = helix.ModelRouter(config)
        fake = FakeBackend()
        router.backend = lambda role: fake

        payload = json.dumps(
            {
                "task": "Build app",
                "active_staged_file": "app.py",
                "staged_files": [
                    {
                        "path": "app.py",
                        "characters": 0,
                    }
                ],
                "recent_history": [],
            }
        )

        role, action = router.action_with_fallback(
            "main",
            "system",
            payload,
        )

        self.assertEqual(role, "main")
        self.assertEqual(
            action["tool"],
            "append_file_raw",
        )
        self.assertEqual(
            fake.calls,
            [True, False],
        )
        self.assertIn(
            'print("hello")',
            action["args"]["_raw_content"],
        )

    def test_model_spec_has_adaptive_raw_generation_defaults(self):
        spec = helix.ModelSpec(
            provider="ollama",
            model="example",
        )

        self.assertEqual(spec.raw_target_chars, 7000)
        self.assertEqual(spec.raw_max_chars, 12000)

    def test_raw_prompt_supports_whole_small_file_generation(self):
        prompt = helix.build_raw_content_prompt(
            original_user=json.dumps({
                "task": "Create a small Python application.",
                "staged_files": [],
            }),
            path="app.py",
            target_chars=7000,
            max_chars=12000,
        )

        self.assertIn(
            "complete remaining file",
            prompt,
        )
        self.assertIn(
            "7000",
            prompt,
        )
        self.assertIn(
            "12000",
            prompt,
        )

    def test_raw_validator_accepts_content_above_old_3500_limit(self):
        router = helix.ModelRouter(
            helix.HelixConfig(
                patchforge_cli="",
                max_steps=40,
                command_timeout_seconds=30,
                models={
                    "main": helix.ModelSpec(
                        provider="ollama",
                        model="example",
                    ),
                },
            )
        )

        content = "x" * 5000

        validated, _ = router._validate_raw_content(
            content,
            max_chars=12000,
        )

        self.assertGreater(len(validated), 3500)

    def test_raw_retry_reduces_target_size(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "int(target_chars * 0.65)",
            source,
        )
        self.assertIn(
            "[target {old_target} -> {target_chars} chars]",
            source,
        )

    def test_raw_generation_adds_final_newline(self):
        class FakeBackend:
            def complete(
                self,
                system,
                user,
                *,
                json_mode=True,
            ):
                return "print('hello')"

        config = helix.HelixConfig(
            patchforge_cli="",
            max_steps=0,
            command_timeout_seconds=10,
            models={
                "main": helix.ModelSpec(
                    provider="ollama",
                    model="main",
                )
            },
        )

        router = helix.ModelRouter(config)

        content = router._complete_raw_content(
            FakeBackend(),
            system="system",
            original_user=json.dumps(
                {
                    "task": "test",
                    "staged_files": [],
                }
            ),
            path="app.py",
        )

        self.assertEqual(
            content,
            "print('hello')\n",
        )

    def test_remaining_requirements_detects_missing_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)

            state = helix.AgentState(
                task="Build an app and add automated tests using pytest.",
                project_root=root,
            )
            state.changed_files.add("main.py")
            state.last_completed_file = "main.py"
            state.post_finish_pending = True

            guidance = agent._remaining_requirements_instruction(state)

            self.assertIn("Automated tests", guidance)
            self.assertIn("Final verification", guidance)
            self.assertIn("main.py", guidance)

    def test_post_finish_blocks_repository_rescan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)

            state = helix.AgentState(
                task="Build app and tests",
                project_root=root,
            )
            state.changed_files.add("main.py")
            state.last_completed_file = "main.py"
            state.post_finish_pending = True

            ok, summary = agent.execute(
                "list_files",
                {},
                state,
            )

            self.assertFalse(ok)
            self.assertIn("Post-finish", summary)
            self.assertIn("main.py", summary)

    def test_post_finish_blocks_immediate_completed_file_reread(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")
            (root / "main.py").write_text(
                "print('ok')\n",
                encoding="utf-8",
            )

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)

            state = helix.AgentState(
                task="Build app and tests",
                project_root=root,
            )
            state.changed_files.add("main.py")
            state.last_completed_file = "main.py"
            state.post_finish_pending = True

            ok, summary = agent.execute(
                "read_file",
                {"path": "main.py"},
                state,
            )

            self.assertFalse(ok)
            self.assertIn("Immediate reread", summary)
            self.assertIn("main.py", summary)

    def test_beginning_next_file_clears_post_finish_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )

            agent = helix.HelixAgent(root, config, prompt)

            state = helix.AgentState(
                task="Build app and tests",
                project_root=root,
            )
            state.last_completed_file = "main.py"
            state.post_finish_pending = True

            ok, _ = agent.execute(
                "begin_file",
                {"path": "test_main.py"},
                state,
            )

            self.assertTrue(ok)
            self.assertFalse(state.post_finish_pending)
            self.assertIn("test_main.py", state.staged_files)

    def test_escalation_handoff_includes_post_finish_remaining_work(self):
        payload = json.dumps(
            {
                "task": "Build a todo app and tests",
                "changed_files": ["main.py"],
                "staged_files": [],
                "active_staged_file": None,
                "last_completed_file": "main.py",
                "post_finish_pending": True,
                "remaining_requirements_instruction": (
                    "Automated tests are still required. "
                    "Final verification has not passed."
                ),
                "last_tool_failure": None,
                "verification_passed": False,
                "recent_history": [],
            }
        )

        handoff = helix.build_escalation_handoff(
            original_user=payload,
            previous_errors=["MAIN stalled after finishing main.py"],
        )

        self.assertIn("post_finish_pending", handoff)
        self.assertIn("Automated tests are still required", handoff)
        self.assertIn("main.py", handoff)
        self.assertIn("MAIN stalled", handoff)

    def test_raw_generation_unwraps_single_markdown_fence(self):
        config = helix.HelixConfig(
            patchforge_cli="",
            max_steps=0,
            command_timeout_seconds=10,
            models={"main": helix.ModelSpec("ollama", "main")},
        )
        router = helix.ModelRouter(config)
        content, unwrapped = router._validate_raw_content("```python\nprint('x')\n```\n")
        self.assertTrue(unwrapped)
        self.assertEqual(content, "print('x')\n")

    def test_staged_recovery_policy_uses_one_repair(self):
        source = Path(helix.__file__).read_text(encoding="utf-8")

        self.assertIn(
            "if staged_repair >= 1:",
            source,
        )
        self.assertIn(
            "[CORE] staged recovery -> finish_file",
            source,
        )
        self.assertIn(
            'f"{staged_repair}/1: "',
            source,
        )

    def test_available_tools_are_state_filtered_in_prompt(self):
        source = Path(helix.__file__).read_text(encoding="utf-8")

        self.assertIn(
            '"available_tools": self._available_tools_for_state(state)',
            source,
        )
        self.assertIn(
            '"append_file_raw",',
            source,
        )
        self.assertIn(
            '"finish_file",',
            source,
        )
        self.assertIn(
            '"list_files",',
            source,
        )

    def test_tool_entry_name_supports_common_schema_shapes(self):
        self.assertEqual(
            helix.HelixAgent._tool_entry_name("read_file"),
            "read_file",
        )
        self.assertEqual(
            helix.HelixAgent._tool_entry_name(
                {"name": "begin_file"}
            ),
            "begin_file",
        )
        self.assertEqual(
            helix.HelixAgent._tool_entry_name(
                {"function": {"name": "run_command"}}
            ),
            "run_command",
        )

    def test_post_finish_test_path_from_todo_file(self):
        state = type(
            "State",
            (),
            {"last_completed_file": "todo.py"},
        )()

        self.assertEqual(
            helix.HelixAgent._post_finish_test_path(state),
            "test_todo.py",
        )

    def test_post_finish_test_path_preserves_directory(self):
        state = type(
            "State",
            (),
            {"last_completed_file": "src/todo_app.py"},
        )()

        self.assertEqual(
            helix.HelixAgent._post_finish_test_path(state),
            "src/test_todo_app.py",
        )

    def test_post_finish_recovery_is_pre_execution(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "_post_finish_action_lock_error(",
            source,
        )
        self.assertIn(
            "[CORE] post-finish recovery -> ",
            source,
        )
        self.assertIn(
            "post-finish-action",
            source,
        )

    def test_staged_python_validation_accepts_valid_python(self):
        error = helix.HelixAgent._staged_python_validation_error(
            "todo.py",
            "def main():\n    return 0\n",
        )

        self.assertIsNone(error)

    def test_staged_python_validation_rejects_truncated_python(self):
        error = helix.HelixAgent._staged_python_validation_error(
            "todo.py",
            "def find_by_id(item_id):\n    if item_id\n",
        )

        self.assertIsNotNone(error)
        self.assertIn(
            "Python syntax validation failed",
            error,
        )

    def test_staged_test_validation_requires_discoverable_test(self):
        error = helix.HelixAgent._staged_python_validation_error(
            "test_todo.py",
            (
                "class TodoManager:\n"
                "    def add(self, text):\n"
                "        return text\n"
            ),
        )

        self.assertIsNotNone(error)
        self.assertIn(
            "Pytest discovery validation failed",
            error,
        )

    def test_staged_test_validation_accepts_pytest_function(self):
        error = helix.HelixAgent._staged_python_validation_error(
            "test_todo.py",
            (
                "def test_add():\n"
                "    assert 1 + 1 == 2\n"
            ),
        )

        self.assertIsNone(error)

    def test_finish_validation_keeps_file_staged(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '"stage": "staged_validation"',
            source,
        )
        self.assertIn(
            "automatic full-file repair attempts.",
            source,
        )

    def test_workspace_can_replace_staged_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = helix.Workspace(Path(tmp))

            workspace.create_file_chunk(
                "example.py",
                "",
                "start",
            )
            workspace.create_file_chunk(
                "example.py",
                "broken",
                "append",
            )

            workspace.replace_staged_file(
                "example.py",
                "print('fixed')\n",
            )

            assembled = workspace.create_file_chunk(
                "example.py",
                "",
                "finish",
            )

            self.assertEqual(
                assembled,
                "print('fixed')\n",
            )

    def test_finish_validation_uses_full_file_repair(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "_repair_staged_python_file(",
            source,
        )
        self.assertIn(
            "HELIX STAGED PYTHON FULL-FILE REPAIR",
            source,
        )
        self.assertIn(
            "replace_staged_file(",
            source,
        )

    def test_test_repair_requires_real_pytest_tests(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "pytest-discoverable test functions",
            source,
        )
        self.assertIn(
            "Do not rewrite the application inside the",
            source,
        )

    def test_raw_prompt_marks_test_file_as_pytest_target(self):
        payload = json.dumps(
            {
                "task": "Build app and add pytest tests.",
                "active_staged_file": "test_todo.py",
                "staged_files": [
                    {
                        "path": "test_todo.py",
                        "characters": 0,
                    }
                ],
                "last_completed_file": "todo.py",
                "changed_files": ["todo.py"],
                "remaining_requirements_instruction": (
                    "Automated tests are still required."
                ),
            }
        )

        prompt = helix.build_raw_content_prompt(
            original_user=payload,
            path="test_todo.py",
        )

        self.assertIn(
            "pytest-discoverable test_* functions",
            prompt,
        )
        self.assertIn(
            "The implementation to test is todo.py",
            prompt,
        )

    def test_empty_staged_recovery_uses_raw_generation(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "[CORE] empty staged recovery ->",
            source,
        )
        self.assertIn(
            'tool = "append_file_raw"',
            source,
        )
        self.assertIn(
            "_complete_raw_content(",
            source,
        )

    def test_post_finish_verification_detects_pytest_q(self):
        state = helix.AgentState(
            task=(
                "Build an application and use pytest -q "
                "as the final verification."
            ),
            project_root=Path("."),
        )
        state.post_finish_pending = True

        self.assertEqual(
            helix.HelixAgent._post_finish_verification_command(
                state
            ),
            "pytest -q",
        )

    def test_post_finish_verification_not_requested_after_pass(self):
        state = helix.AgentState(
            task="Use pytest -q as the final verification.",
            project_root=Path("."),
        )
        state.post_finish_pending = True
        state.verification_passed = True

        self.assertEqual(
            helix.HelixAgent._post_finish_verification_command(
                state
            ),
            "",
        )

    def test_post_finish_recovery_can_force_verification(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "[CORE] post-finish recovery ->",
            source,
        )
        self.assertIn(
            '"verification": True',
            source,
        )
        self.assertIn(
            "POST-FINISH ACTION LOCK: final verification is now ",
            source,
        )
        self.assertIn(
            "required. Run exactly",
            source,
        )

    def test_staged_action_lock_rejects_unrelated_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )
            agent = helix.HelixAgent(root, config, prompt)

            state = helix.AgentState(
                task="Create tests",
                project_root=root,
            )
            state.staged_files["test_app.py"] = 0

            error = agent._staged_action_lock_error(
                "list_files",
                {},
                state,
            )

            self.assertIsNotNone(error)
            self.assertIn("STAGED WORK LOCK", error)
            self.assertIn("test_app.py", error)

    def test_empty_staged_file_only_allows_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("prompt\n", encoding="utf-8")

            config = helix.HelixConfig(
                patchforge_cli="",
                max_steps=0,
                command_timeout_seconds=10,
                models={"main": helix.ModelSpec("ollama", "main")},
            )
            agent = helix.HelixAgent(root, config, prompt)

            state = helix.AgentState(
                task="Create tests",
                project_root=root,
            )
            state.staged_files["test_app.py"] = 0

            self.assertIsNone(
                agent._staged_action_lock_error(
                    "append_file_raw",
                    {"path": "test_app.py"},
                    state,
                )
            )

            self.assertIsNotNone(
                agent._staged_action_lock_error(
                    "finish_file",
                    {"path": "test_app.py"},
                    state,
                )
            )

    def test_v060_architect_state_exists(self):
        state = helix.AgentState(
            task="test",
            project_root=Path("."),
        )

        self.assertEqual(
            state.architect_recoveries,
            0,
        )
        self.assertEqual(
            state.architect_contract,
            "",
        )

    def test_v060_new_roles_use_orchestrator(self):
        config = helix.HelixConfig(
            patchforge_cli="",
            max_steps=40,
            command_timeout_seconds=30,
            models={
                "architect": helix.ModelSpec(
                    provider="ollama",
                    model="gpt-oss:20b",
                ),
                "orchestrator": helix.ModelSpec(
                    provider="ollama",
                    model="qwen3.5:4b",
                ),
                "coder": helix.ModelSpec(
                    provider="ollama",
                    model="qwen3.5:9b",
                ),
            },
        )

        router = helix.ModelRouter(config)

        state = helix.AgentState(
            task="Build something.",
            project_root=Path("."),
        )

        self.assertEqual(
            router.choose_role(state),
            "orchestrator",
        )

    def test_v060_coder_not_action_fallback(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'for fallback in ("orchestrator", "remote")',
            source,
        )

    def test_v060_architect_plan_is_persisted(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "development_plan.md",
            source,
        )
        self.assertIn(
            "HELIX_EXECUTION_CONTRACT_START",
            source,
        )

    def test_v060_architect_recovery_is_persisted(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "recovery-{recovery_number:03d}.json",
            source,
        )
        self.assertIn(
            "recovery-{recovery_number:03d}.md",
            source,
        )

    def test_v060_architect_unloads_normal_models(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '("orchestrator", "coder", "remote")',
            source,
        )
        self.assertIn(
            '"keep_alive": 0',
            source,
        )

    def test_v060_benchmark_supports_role_overrides(self):
        parser = helix.build_parser()

        args = parser.parse_args([
            "--benchmark",
            "benchmarks/todo.json",
            "--benchmark-model",
            "coder-model",
            "--benchmark-orchestrator-model",
            "planner-model",
            "--benchmark-architect-model",
            "architect-model",
        ])

        self.assertEqual(
            args.benchmark_model,
            "coder-model",
        )
        self.assertEqual(
            args.benchmark_orchestrator_model,
            "planner-model",
        )
        self.assertEqual(
            args.benchmark_architect_model,
            "architect-model",
        )

    def test_v061_architect_prompt_has_requirement_provenance(self):
        prompt_path = (
            Path(helix.__file__).parent
            / "prompts"
            / "architect.txt"
        )

        prompt = prompt_path.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "EXPLICIT USER REQUIREMENT",
            prompt,
        )
        self.assertIn(
            "DERIVED CORRECTNESS REQUIREMENT",
            prompt,
        )
        self.assertIn(
            "ARCHITECTURE / DESIGN RECOMMENDATION",
            prompt,
        )
        self.assertIn(
            "OPTIONAL ENHANCEMENT",
            prompt,
        )

    def test_v061_execution_contract_rejects_design_preferences(self):
        prompt_path = (
            Path(helix.__file__).parent
            / "prompts"
            / "architect.txt"
        )

        prompt = prompt_path.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "NEVER invent a user requirement",
            prompt,
        )
        self.assertIn(
            "It MUST NOT contain:",
            prompt,
        )
        self.assertIn(
            "invented package names",
            prompt,
        )
        self.assertIn(
            "invented storage paths",
            prompt,
        )

    def test_v061_architect_guards_process_instruction_semantics(self):
        prompt_path = (
            Path(helix.__file__).parent
            / "prompts"
            / "architect.txt"
        )

        prompt = prompt_path.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "Do not stop until the application works",
            prompt,
        )
        self.assertIn(
            "does NOT mean the generated application",
            prompt,
        )

    def test_v061_architect_guards_list_semantics(self):
        prompt_path = (
            Path(helix.__file__).parent
            / "prompts"
            / "architect.txt"
        )

        prompt = prompt_path.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "list",
            prompt,
        )
        self.assertIn(
            "list --all",
            prompt,
        )
        self.assertIn(
            "That changes the requested default behavior",
            prompt,
        )

    def test_v061_architect_requires_provenance_table(self):
        prompt_path = (
            Path(helix.__file__).parent
            / "prompts"
            / "architect.txt"
        )

        prompt = prompt_path.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "REQUIREMENT PROVENANCE TABLE",
            prompt,
        )
        self.assertIn(
            "| ID | Requirement / Decision | Provenance | Binding | Reason |",
            prompt,
        )

    def test_v062_ollama_tracks_response_metadata(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "last_response_metadata",
            source,
        )
        self.assertIn(
            '"thinking_chars"',
            source,
        )
        self.assertIn(
            '"content_chars"',
            source,
        )
        self.assertIn(
            '"done_reason"',
            source,
        )

    def test_v062_architect_retries_empty_response(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "max_empty_attempts = 2",
            source,
        )
        self.assertIn(
            "empty-response ",
            source,
        )
        self.assertIn(
            "retry 1/1...",
            source,
        )
        self.assertIn(
            "after 2 attempts",
            source,
        )

    def test_v062_architect_has_progress_heartbeat(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "helix-architect-heartbeat",
            source,
        )
        self.assertIn(
            "waiting for model...",
            source,
        )
        self.assertIn(
            "stop_heartbeat.wait(10.0)",
            source,
        )

    def test_v062_architect_reports_response_size(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "response received",
            source,
        )
        self.assertIn(
            "execution contract",
            source,
        )

    def test_v062_thinking_is_not_used_as_plan(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'thinking = str(',
            source,
        )

        self.assertIn(
            "return content",
            source,
        )

        self.assertNotIn(
            "return thinking",
            source,
        )

    def test_v063_model_spec_has_num_predict(self):
        spec = helix.ModelSpec(
            provider="ollama",
            model="example",
        )

        self.assertEqual(
            spec.num_predict,
            4096,
        )

    def test_v063_ollama_uses_num_predict(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '"num_predict": self.spec.num_predict',
            source,
        )

    def test_v063_architect_supports_continuation(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'done_reason != "length"',
            source,
        )
        self.assertIn(
            "ARCHITECT CONTINUATION REQUIRED",
            source,
        )
        self.assertIn(
            "requesting continuation",
            source,
        )

    def test_v063_architect_has_plan_limit(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "max_plan_chars = 60000",
            source,
        )
        self.assertIn(
            "max_continuations = 5",
            source,
        )

    def test_v063_global_empty_response_diagnostics(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "content_chars=",
            source,
        )
        self.assertIn(
            "thinking_chars=",
            source,
        )
        self.assertIn(
            "json_mode=True",
            source,
        )
        self.assertIn(
            "json_mode=False",
            source,
        )

    def test_v064_model_spec_has_num_ctx(self):
        spec = helix.ModelSpec(
            provider="ollama",
            model="example",
        )

        self.assertEqual(
            spec.num_ctx,
            8192,
        )

    def test_v064_ollama_uses_num_ctx(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '"num_ctx": self.spec.num_ctx',
            source,
        )

    def test_v064_architect_uses_smaller_continuation_tail(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "previous_tail = content[-3000:]",
            source,
        )
        self.assertNotIn(
            "previous_tail = content[-8000:]",
            source,
        )

    def test_v064_architect_reports_prompt_and_context(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "prompt_eval_count = metadata.get(",
            source,
        )
        self.assertIn(
            'f"context={getattr(backend.spec, \'num_ctx\', \'?\')}"',
            source,
        )

    def test_v064_benchmark_preserves_role_context(self):
        source = Path(helix.__file__).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "existing.num_ctx",
            source,
        )
        self.assertIn(
            "old_orchestrator.num_ctx",
            source,
        )
        self.assertIn(
            "old_architect.num_ctx",
            source,
        )

    def test_model_circuit_breaker_marks_role_unhealthy(self):
        config = helix.HelixConfig(
            patchforge_cli="",
            max_steps=0,
            command_timeout_seconds=10,
            models={
                "main": helix.ModelSpec("ollama", "main"),
                "large": helix.ModelSpec("ollama", "large"),
            },
            model_circuit_breaker_failures=3,
        )

        router = helix.ModelRouter(config)

        router._record_model_failure("large")
        router._record_model_failure("large")
        self.assertTrue(router._enabled("large"))

        router._record_model_failure("large")
        self.assertFalse(router._enabled("large"))

        router.reset_task_health()
        self.assertTrue(router._enabled("large"))

    def test_load_benchmark_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "tiny",
                        "task": "Create hello.py",
                        "verification_command": "python hello.py",
                    }
                ),
                encoding="utf-8",
            )

            spec = helix.load_benchmark_spec(path)

            self.assertEqual(spec["name"], "tiny")
            self.assertEqual(spec["task"], "Create hello.py")

    def test_parser_accepts_benchmark_options(self):
        args = helix.build_parser().parse_args(
            [
                "--benchmark",
                "bench.json",
                "--benchmark-model",
                "qwen3.5:9b",
                "--runs",
                "5",
            ]
        )

        self.assertEqual(args.benchmark, "bench.json")
        self.assertEqual(
            args.benchmark_model,
            "qwen3.5:9b",
        )
        self.assertEqual(args.runs, 5)

    def test_parser_accepts_benchmark_progress(self):
        args = helix.build_parser().parse_args(
            [
                "--benchmark",
                "bench.json",
                "--benchmark-model",
                "qwen3.5:9b",
                "--benchmark-progress",
            ]
        )
        self.assertTrue(args.benchmark_progress)

    def test_benchmark_tee_writes_to_both_streams(self):
        first = io.StringIO()
        second = io.StringIO()
        tee = helix._BenchmarkTee(first, second)

        tee.write("hello")
        tee.flush()

        self.assertEqual(first.getvalue(), "hello")
        self.assertEqual(second.getvalue(), "hello")

    def test_patchforge_detects_windows_encoding_failure(self):
        result = {
            "ok": False,
            "stderr": (
                "UnicodeEncodeError: 'charmap' codec can't encode "
                "character using cp1252"
            ),
        }

        self.assertTrue(
            helix.PatchForgeAdapter.infrastructure_failure(result)
        )

    def test_patchforge_normal_failure_is_not_infrastructure_failure(self):
        result = {
            "ok": False,
            "stderr": "Patch anchor did not match.",
        }

        self.assertFalse(
            helix.PatchForgeAdapter.infrastructure_failure(result)
        )

    def test_interruptible_call_returns_value(self):
        self.assertEqual(
            helix.interruptible_call(lambda: "ok"),
            "ok",
        )

if __name__ == "__main__":
    unittest.main()
