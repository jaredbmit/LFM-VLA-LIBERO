"""CALVIN evaluation client: runs the env and calls the VLA inference server for actions.

Run in the calvin_venv conda env. Connects to eval_server.py over TCP.

Usage:
    conda activate calvin_venv
    python scripts/eval_client.py \
        --dataset_path ~/drl/calvin/dataset/task_D_D \
        --num_sequences 1000
"""

import argparse
import base64
import io
import json
import os
import socket
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import count_success, get_env_state_for_initial_condition
from calvin_env.envs.play_table_env import get_env

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

EP_LEN = 360


class InferenceClient:
    """Connects to the VLA inference server and manages action chunking."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None
        self.buf = bytearray()
        self.action_queue = []

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"Connected to inference server at {self.host}:{self.port}")

    def close(self):
        if self.sock:
            self.sock.close()

    def shutdown_server(self):
        self._send({"shutdown": True})
        self.close()

    def reset(self):
        """Clear action buffer at the start of each subtask."""
        self.action_queue = []

    def step(self, obs, instruction: str) -> np.ndarray:
        """Get the next action, querying the server when the buffer is empty."""
        if not self.action_queue:
            image_rgb = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3) uint8
            actions = self._query(image_rgb, instruction)
            self.action_queue = actions

        return np.array(self.action_queue.pop(0))

    def _query(self, image_rgb: np.ndarray, instruction: str) -> list[list[float]]:
        buf = io.BytesIO()
        Image.fromarray(image_rgb).save(buf, format="JPEG", quality=95)
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        request = {"image": image_b64, "instruction": instruction}
        self._send(request)
        response = self._recv()
        return response["actions"]

    def _send(self, data: dict):
        self.sock.sendall(json.dumps(data).encode() + b"\n")

    def _recv(self) -> dict:
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("Server closed connection")
            self.buf.extend(chunk)
        idx = self.buf.index(b"\n")
        line = self.buf[:idx].decode("utf-8")
        del self.buf[:idx + 1]
        return json.loads(line)


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    return get_env(val_folder, show_gui=False)


def rollout(env, client, task_oracle, subtask, val_annotations, debug):
    if debug:
        print(f"{subtask} ", end="", flush=True)

    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    client.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        action = client.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print("success ", end="", flush=True)
            return True

    if debug:
        print("fail ", end="", flush=True)
    return False


def evaluate_sequence(env, client, task_oracle, initial_state, eval_sequence, val_annotations, debug):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        print(f"\nEvaluating: {' -> '.join(eval_sequence)}")
        print("  ", end="")

    for subtask in eval_sequence:
        success = rollout(env, client, task_oracle, subtask, val_annotations, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def print_and_save(results, sequences, log_dir, num_sequences):
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}

    print(f"\nResults ({num_sequences} sequences):")
    print(f"  Average successful sequence length: {avg_seq_len:.2f}")
    print("  Chain success rates:")
    for i, sr in chain_sr.items():
        print(f"    {i}/5: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()
    for result, (_, sequence) in zip(results, sequences):
        for task in sequence[:result]:
            cnt_success[task] += 1
        if result < len(sequence):
            cnt_fail[sequence[result]] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    print("  Per-task success rates:")
    for task in sorted(total):
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"    {task}: {cnt_success[task]}/{total[task]} ({cnt_success[task]/total[task]*100:.1f}%)")

    data = {
        "avg_seq_len": avg_seq_len,
        "chain_sr": chain_sr,
        "task_info": task_info,
    }

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "results.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {log_dir / 'results.json'}")

    return data


def main():
    seed_everything(0, workers=True)

    parser = argparse.ArgumentParser(description="CALVIN evaluation client")
    parser.add_argument("--dataset_path", required=True, help="Path to CALVIN dataset (e.g. task_D_D)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--eval_log_dir", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load CALVIN task oracle and annotations
    for candidate in [
        Path.home() / "drl/calvin/calvin_models/conf",
        Path("/home/jared/drl/calvin/calvin_models/conf"),
    ]:
        if (candidate / "callbacks/rollout/tasks/new_playtable_tasks.yaml").exists():
            calvin_conf_dir = candidate
            break
    else:
        raise FileNotFoundError("Could not find CALVIN conf directory")

    task_cfg = OmegaConf.load(calvin_conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(calvin_conf_dir / "annotations/new_playtable_validation.yaml")

    print(f"Creating CALVIN environment from {args.dataset_path}...")
    env = make_env(args.dataset_path)

    print(f"Generating {args.num_sequences} evaluation sequences...")
    eval_sequences = get_sequences(args.num_sequences)

    client = InferenceClient(args.host, args.port)
    client.connect()

    results = []
    seq_iter = eval_sequences
    if not args.debug:
        seq_iter = tqdm(eval_sequences, position=0, leave=True)

    start_time = time.time()

    try:
        for initial_state, eval_sequence in seq_iter:
            result = evaluate_sequence(
                env, client, task_oracle, initial_state, eval_sequence, val_annotations, args.debug,
            )
            results.append(result)
            if not args.debug:
                seq_iter.set_description(
                    " ".join([f"{i+1}/5: {v*100:.1f}% |" for i, v in enumerate(count_success(results))])
                )
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(results)} sequences.")
    finally:
        elapsed = time.time() - start_time
        print(f"\nEvaluation took {elapsed:.0f}s ({elapsed/3600:.1f}h)")

        if results:
            print_and_save(results, eval_sequences[:len(results)], args.eval_log_dir, len(results))

        client.close()


if __name__ == "__main__":
    main()
