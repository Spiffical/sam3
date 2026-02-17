#!/usr/bin/env python3
"""
Run model comparison matrix for a single video using SAM3 agent workflow.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    default_output_root = os.path.join(
        os.environ.get("PROJECT_ROOT", "nibi_model_compare"), "runs", "pilot"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_file",
        default=os.path.join("nibi_model_compare", "models.json"),
        type=str,
    )
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument(
        "--prompt", default="segment all visible marine organisms", type=str
    )
    parser.add_argument(
        "--output_root",
        default=default_output_root,
        type=str,
    )
    parser.add_argument("--gpus", default="0,1,2,3", type=str)
    parser.add_argument("--python_exec", default=sys.executable, type=str)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_launch", action="store_true")
    parser.add_argument("--allow_disabled", action="store_true")
    parser.add_argument("--continue_on_failure", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--wait_timeout_sec", default=900, type=int)
    return parser.parse_args()


def read_models(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    models = payload.get("models", [])
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError(f"No models found in {path}")
    return models


def wait_for_openai_models_endpoint(server_url: str, timeout_sec: int) -> None:
    base = server_url.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/models"
    else:
        url = f"{base}/v1/models"
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}. Last error: {last_error}")


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> int:
    print(f"[cmd] {' '.join(cmd)}")
    return subprocess.run(cmd, env=env, cwd=ROOT).returncode


def run_single_model(
    model_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    model_key = model_cfg.get("key", model_cfg.get("model_id", "unknown_model"))
    runner = model_cfg.get("runner")
    run_dir = os.path.join(args.output_root, model_key)
    os.makedirs(run_dir, exist_ok=True)

    launcher_metrics: dict[str, Any] = {
        "model_key": model_key,
        "runner": runner,
        "model_id": model_cfg.get("model_id"),
        "status": "started",
        "run_dir": os.path.abspath(run_dir),
    }

    run_metrics_path = os.path.join(run_dir, "run_metrics.json")
    if args.skip_existing and os.path.exists(run_metrics_path):
        prior = read_json(run_metrics_path)
        if prior.get("status", "").startswith("success"):
            launcher_metrics["status"] = "skipped_existing_success"
            return 0, launcher_metrics

    launch_proc: subprocess.Popen[str] | None = None
    env = os.environ.copy()
    api_key_env = model_cfg.get("api_key_env")
    api_key = env.get(api_key_env, "") if api_key_env else ""
    start = time.time()

    try:
        if runner == "openai_agent":
            if not args.no_launch and model_cfg.get("launch_cmd"):
                print(f"[launch] {model_cfg['launch_cmd']}")
                launch_proc = subprocess.Popen(
                    model_cfg["launch_cmd"],
                    shell=True,
                    cwd=ROOT,
                    executable="/bin/bash",
                )
                wait_for_openai_models_endpoint(
                    model_cfg["server_url"],
                    timeout_sec=args.wait_timeout_sec,
                )

            cmd = [
                args.python_exec,
                os.path.join("nibi_model_compare", "run_video_agent_openai.py"),
                "--video_path",
                args.video_path,
                "--prompt",
                args.prompt,
                "--server_url",
                model_cfg["server_url"],
                "--model",
                model_cfg["model_id"],
                "--output_dir",
                run_dir,
                "--gpus",
                args.gpus,
                "--max_completion_tokens",
                str(model_cfg.get("max_completion_tokens", 4096)),
            ]
            if api_key:
                cmd.extend(["--api_key", api_key])
            if args.debug:
                cmd.append("--debug")
            if args.save_prompts:
                cmd.append("--save_prompts")
            rc = run_cmd(cmd, env=env)

        elif runner == "gemini_agent":
            cmd = [
                args.python_exec,
                os.path.join("sam3", "apps", "gemini_video_agent.py"),
                "--video_path",
                args.video_path,
                "--prompt",
                args.prompt,
                "--output_dir",
                run_dir,
                "--gpus",
                args.gpus,
            ]
            if api_key:
                cmd.extend(["--api_key", api_key])
            rc = run_cmd(cmd, env=env)

        else:
            raise ValueError(f"Unsupported runner: {runner}")

        launcher_metrics["status"] = "success" if rc == 0 else "failed"
        launcher_metrics["return_code"] = rc

    except Exception as exc:
        launcher_metrics["status"] = "failed_exception"
        launcher_metrics["error"] = str(exc)
        rc = 1

    finally:
        if launch_proc is not None:
            launch_proc.send_signal(signal.SIGTERM)
            try:
                launch_proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                launch_proc.kill()
        launcher_metrics["runtime_sec"] = round(time.time() - start, 3)
        launcher_metrics["finished_at_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        write_json(os.path.join(run_dir, "launcher_metrics.json"), launcher_metrics)

    return rc, launcher_metrics


def maybe_run_summary(args: argparse.Namespace) -> None:
    summary_cmd = [
        args.python_exec,
        os.path.join("nibi_model_compare", "summarize_runs.py"),
        "--output_root",
        args.output_root,
    ]
    run_cmd(summary_cmd)

    paste_cmd = [
        args.python_exec,
        os.path.join("nibi_model_compare", "generate_paste_report.py"),
        "--summary_csv",
        os.path.join(args.output_root, "summary.csv"),
        "--output_path",
        os.path.join(args.output_root, "PASTE_TO_CODEX.md"),
    ]
    run_cmd(paste_cmd)


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    models = read_models(args.models_file)
    failed: list[str] = []
    completed: list[str] = []
    skipped: list[str] = []

    for model_cfg in models:
        enabled = bool(model_cfg.get("enabled", True))
        model_key = model_cfg.get("key", model_cfg.get("model_id", "unknown_model"))
        if not enabled and not args.allow_disabled:
            print(f"[skip-disabled] {model_key}")
            skipped.append(model_key)
            continue

        rc, info = run_single_model(model_cfg, args)
        if info.get("status") == "skipped_existing_success":
            skipped.append(model_key)
            continue

        if rc == 0:
            completed.append(model_key)
        else:
            failed.append(model_key)
            if not args.continue_on_failure:
                break

    maybe_run_summary(args)

    print("")
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
