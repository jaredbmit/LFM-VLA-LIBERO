"""CALVIN evaluation client: runs the env and calls the VLA inference server for actions.

Run in the calvin_venv conda env. Connects to eval_server.py over TCP.

With --num_workers > 1, spawns N independent processes each running their own
CALVIN env. All workers connect to the same server, which batches their requests
for efficient GPU utilization.

Usage:
    conda activate calvin_venv
    python scripts/eval_client.py \
        --dataset_path ~/drl/calvin/dataset/task_D_D \
        --num_sequences 1000 \
        --num_workers 4
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import multiprocessing as mp
import os
import socket
import time
from collections import Counter
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

    def connect(self, retry_interval: float = 5.0, timeout: float = 300.0):
        deadline = time.time() + timeout
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                print(f"Worker {os.getpid()}: connected to inference server at {self.host}:{self.port}")
                return
            except ConnectionRefusedError:
                self.sock.close()
                if time.time() >= deadline:
                    raise ConnectionRefusedError(
                        f"Could not connect to inference server at {self.host}:{self.port} "
                        f"after {timeout:.0f}s. Is eval_server.py running?"
                    )
                print(f"Waiting for inference server at {self.host}:{self.port} ...", flush=True)
                time.sleep(retry_interval)

    def close(self):
        if self.sock:
            self.sock.close()

    def shutdown_server(self):
        self._send({"shutdown": True})
        self.close()

    def reset(self):
        self.action_queue = []

    def step(self, obs, instruction: str) -> np.ndarray:
        if not self.action_queue:
            image_rgb = obs["rgb_obs"]["rgb_static"]
            actions = self._query(image_rgb, instruction)
            self.action_queue = actions
        return np.array(self.action_queue.pop(0))

    def _query(self, image_rgb: np.ndarray, instruction: str) -> list[list[float]]:
        buf = io.BytesIO()
        Image.fromarray(image_rgb).save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        self._send({"image": image_b64, "instruction": instruction})
        return self._recv()["actions"]

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


def _annotate_frame(frame: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(frame).resize((frame.shape[1] * 2, frame.shape[0] * 2), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, img.height - 18, img.width, img.height], fill=(0, 0, 0))
    draw.text((4, img.height - 16), label, fill=(255, 255, 255), font=font)
    return np.array(img)


def rollout(env, client, task_oracle, subtask, val_annotations, debug, frames=None):
    if debug:
        print(f"{subtask} ", end="", flush=True)

    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    client.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        if frames is not None:
            frames.append(_annotate_frame(obs["rgb_obs"]["rgb_static"], lang_annotation))

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


def evaluate_sequence(env, client, task_oracle, initial_state, eval_sequence, val_annotations, debug, frames=None):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        print(f"\nEvaluating: {' -> '.join(eval_sequence)}")
        print("  ", end="")

    for subtask in eval_sequence:
        success = rollout(env, client, task_oracle, subtask, val_annotations, debug, frames=frames)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def save_video(frames: list, path: Path, fps: int = 30):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"  Saved video: {path}")


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


def _worker_main(worker_id: int, sequences: list, args_dict: dict, result_queue: mp.Queue):
    """Entry point for each worker process."""
    try:
        _worker_main_inner(worker_id, sequences, args_dict, result_queue)
    except Exception as e:
        import traceback
        print(f"Worker {worker_id} crashed: {e}", flush=True)
        traceback.print_exc()
        result_queue.put((worker_id, None, None))  # signal done


def _worker_main_inner(worker_id: int, sequences: list, args_dict: dict, result_queue: mp.Queue):
    seed_everything(worker_id, workers=True)

    args = argparse.Namespace(**args_dict)

    calvin_conf_dir = None
    for candidate in [
        Path.home() / "drl/calvin/calvin_models/conf",
        Path("/home/jared/drl/calvin/calvin_models/conf"),
    ]:
        if (candidate / "callbacks/rollout/tasks/new_playtable_tasks.yaml").exists():
            calvin_conf_dir = candidate
            break
    if calvin_conf_dir is None:
        raise FileNotFoundError("Could not find CALVIN conf directory")

    task_cfg = OmegaConf.load(calvin_conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(calvin_conf_dir / "annotations/new_playtable_validation.yaml")

    env = make_env(args.dataset_path)
    client = InferenceClient(args.host, args.port)
    client.connect()

    for seq_idx, (initial_state, eval_sequence) in enumerate(sequences):
        result = evaluate_sequence(
            env, client, task_oracle, initial_state, eval_sequence,
            val_annotations, args.debug,
        )
        result_queue.put((worker_id, seq_idx, result))

    client.close()
    result_queue.put((worker_id, None, None))  # sentinel: worker done


def main():
    seed_everything(0, workers=True)

    parser = argparse.ArgumentParser(description="CALVIN evaluation client")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel env worker processes (default: 1)")
    parser.add_argument("--eval_log_dir", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_videos", type=int, default=0)
    parser.add_argument("--video_dir", default=None)
    args = parser.parse_args()

    eval_sequences = get_sequences(args.num_sequences)

    if args.num_workers == 1:
        # Single-process path — unchanged behavior
        calvin_conf_dir = None
        for candidate in [
            Path.home() / "drl/calvin/calvin_models/conf",
            Path("/home/jared/drl/calvin/calvin_models/conf"),
        ]:
            if (candidate / "callbacks/rollout/tasks/new_playtable_tasks.yaml").exists():
                calvin_conf_dir = candidate
                break
        if calvin_conf_dir is None:
            raise FileNotFoundError("Could not find CALVIN conf directory")

        task_cfg = OmegaConf.load(calvin_conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        task_oracle = hydra.utils.instantiate(task_cfg)
        val_annotations = OmegaConf.load(calvin_conf_dir / "annotations/new_playtable_validation.yaml")

        print(f"Creating CALVIN environment from {args.dataset_path}...")
        env = make_env(args.dataset_path)

        client = InferenceClient(args.host, args.port)
        client.connect()

        video_dir = None
        if args.num_videos > 0:
            video_dir = Path(args.video_dir) if args.video_dir else (
                Path(args.eval_log_dir) / "videos" if args.eval_log_dir else Path("eval_videos")
            )

        results = []
        seq_iter = eval_sequences if args.debug else tqdm(eval_sequences)
        start_time = time.time()
        try:
            for seq_idx, (initial_state, eval_sequence) in enumerate(seq_iter):
                frames = [] if (video_dir and seq_idx < args.num_videos) else None
                result = evaluate_sequence(
                    env, client, task_oracle, initial_state, eval_sequence,
                    val_annotations, args.debug, frames=frames,
                )
                results.append(result)
                if frames:
                    tasks_str = "_".join(eval_sequence[:result + 1] if result < len(eval_sequence) else eval_sequence)
                    save_video(frames, video_dir / f"seq{seq_idx:03d}_{result}of{len(eval_sequence)}_{tasks_str[:60]}.mp4")
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

    else:
        # Multi-process path: split sequences across workers
        n = args.num_workers
        shards = [eval_sequences[i::n] for i in range(n)]
        print(f"Launching {n} worker processes ({len(eval_sequences)} sequences split ~{len(eval_sequences)//n} each)")

        args_dict = vars(args)
        result_queue: mp.Queue = mp.Queue()
        processes = []
        start_time = time.time()

        for worker_id, shard in enumerate(shards):
            p = mp.Process(target=_worker_main,
                           args=(worker_id, shard, args_dict, result_queue),
                           daemon=True)
            p.start()
            processes.append(p)

        # Collect per-sequence results with progress bar
        results = [None] * len(eval_sequences)
        completed = []
        workers_done = 0
        pbar = tqdm(total=len(eval_sequences), disable=args.debug)
        try:
            while workers_done < n:
                worker_id, seq_idx, result = result_queue.get(timeout=7200)
                if seq_idx is None:
                    # Sentinel: worker finished
                    workers_done += 1
                    continue
                global_idx = worker_id + seq_idx * n
                results[global_idx] = result
                completed.append(result)
                pbar.update(1)
                pbar.set_description(
                    " ".join([f"{i+1}/5: {v*100:.1f}% |"
                              for i, v in enumerate(count_success(completed))])
                )
        except Exception as e:
            print(f"Error collecting results: {e}")
        finally:
            pbar.close()
            for p in processes:
                p.join(timeout=5.0)

        results = [r for r in results if r is not None]

        elapsed = time.time() - start_time
        print(f"\nEvaluation took {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        if results:
            print_and_save(results, eval_sequences[:len(results)], args.eval_log_dir, len(results))


if __name__ == "__main__":
    main()
