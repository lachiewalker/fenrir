"""Generated training samples, written once to disk and read for many epochs.

Generating one sample takes most of a second of CPU, so training on fresh
samples leaves the GPU idle. `cache.py` writes samples in shards on every
core, and `ShardDataset` reads them back in the form the generative
datasets yield.

Images are stored as uint8 after min-max scaling. The reader restores the
0..1 range, and subtracts the mean for the tasks that train on centred input.
"""

import multiprocessing as mp
import os
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, TensorDataset, get_worker_info

TASKS = ("position", "rotation", "quad")
SHARD = 250


def _centred(task: str) -> bool:
    return task in ("position", "rotation")


def _dataset(task: str):
    if task == "position":
        from src.fen_recognition.dataset import GenerativeBoardDataset
        return GenerativeBoardDataset(game="chess", augment_ratio=0.8, affine_augment_ratio=0.8)
    if task == "rotation":
        from src.board_image_rotation.dataset import GenerativeRotationDataset
        return GenerativeRotationDataset(game="chess")
    from src.bounding_box.dataset import GenerativeBboxDataset
    return GenerativeBboxDataset(game="chess")


def _to_u8(img: torch.Tensor) -> torch.Tensor:
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return torch.zeros_like(img, dtype=torch.uint8)
    return ((img - lo) / (hi - lo) * 255).round().to(torch.uint8)


def _write_part(task: str, out: str, worker: int, count: int, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    it = iter(_dataset(task))
    images, targets = [], []
    shard = 0
    for _ in range(count):
        sample = next(it)
        images.append(_to_u8(sample[0]))
        if task == "position":
            targets.append(sample[1].argmax(dim=1).to(torch.uint8))
        elif task == "rotation":
            targets.append(torch.tensor(sample[1], dtype=torch.uint8))
        else:
            targets.append(sample[1])
        if len(images) == SHARD:
            torch.save({"images": torch.stack(images), "targets": torch.stack(targets)},
                       Path(out) / f"{worker:02d}-{shard:04d}.pt")
            images, targets = [], []
            shard += 1
    if images:
        torch.save({"images": torch.stack(images), "targets": torch.stack(targets)},
                   Path(out) / f"{worker:02d}-{shard:04d}.pt")


def write(task: str, out: Path, samples: int, workers: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    per = -(-samples // workers)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_write_part, args=(task, str(out), w, per, seed * 1000 + w)) for w in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"a cache worker for {task} failed")


def _restore(task: str, u8: torch.Tensor, target: torch.Tensor, channels: int = 13):
    img = u8.float() / 255.0
    if _centred(task):
        img = img - img.mean()
    if task == "position":
        return img, torch.nn.functional.one_hot(target.long(), channels).float()
    if task == "rotation":
        return img, int(target)
    from src.bounding_box.dataset import corners_to_mask
    return img, target, corners_to_mask(target)


class ShardDataset(IterableDataset):
    """Cycles through the shards in `root` without end, in random order."""

    def __init__(self, root: Path, task: str):
        self.files = sorted(Path(root).glob("*.pt"))
        self.task = task
        if not self.files:
            raise FileNotFoundError(f"No shards in {root}")

    def __iter__(self):
        info = get_worker_info()
        files = list(self.files)
        if info is not None and len(files) >= info.num_workers:
            files = files[info.id::info.num_workers]
        rng = random.Random(os.getpid())
        while True:
            rng.shuffle(files)
            for f in files:
                data = torch.load(f)
                order = list(range(len(data["images"])))
                rng.shuffle(order)
                for i in order:
                    yield _restore(self.task, data["images"][i], data["targets"][i])


def load_test(root: Path, task: str, limit: int):
    samples = []
    for f in sorted(Path(root).glob("*.pt")):
        data = torch.load(f)
        for u8, target in zip(data["images"], data["targets"]):
            samples.append(_restore(task, u8, target))
            if len(samples) >= limit:
                break
        if len(samples) >= limit:
            break
    if task == "quad":
        return samples
    images = torch.stack([s[0] for s in samples])
    targets = torch.stack([s[1] if torch.is_tensor(s[1]) else torch.tensor(s[1]) for s in samples])
    return TensorDataset(images, targets)


def train_and_test(task: str):
    """The cached train set and test set named by FENRIR_CACHE, or `None`."""
    root = os.environ.get("FENRIR_CACHE")
    if not root:
        return None
    root = Path(root) / task
    limit = int(os.environ.get("FENRIR_TEST_SIZE", 1000))
    return ShardDataset(root / "train", task), load_test(root / "test", task, limit)
