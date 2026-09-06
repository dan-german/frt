from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from frt.ml.dataset import TranscriptionDataset, collate_transcription_batch
from frt.ml.features import DEFAULT_HOP_LENGTH
from frt.ml.model import OnsetsAndFramesModel


def masked_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = 1.0 + targets * (pos_weight - 1.0)
    mask_expanded = mask[:, :, None, None].expand_as(loss)
    masked_loss = loss * weights * mask_expanded
    return masked_loss.sum() / mask_expanded.sum().clamp_min(1)


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = torch.sigmoid(logits) >= threshold
    target_active = targets >= 0.5
    mask_expanded = mask[:, :, None, None].expand_as(predictions)
    predictions = predictions & mask_expanded
    target_active = target_active & mask_expanded

    true_positive = (predictions & target_active).sum().item()
    false_positive = (predictions & ~target_active).sum().item()
    false_negative = (~predictions & target_active).sum().item()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def split_indices(
    count: int,
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if count <= 0:
        raise ValueError("dataset must contain at least one example")
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0.0, 1.0)")

    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(count * val_split))
    if count > 1 and val_split > 0:
        val_count = min(max(val_count, 1), count - 1)
    validation = indices[:val_count]
    training = indices[val_count:]
    return training, validation


def move_batch(
    batch: dict,
    device: torch.device,
) -> dict:
    return {
        **batch,
        "features": batch["features"].to(device),
        "frame_targets": batch["frame_targets"].to(device),
        "onset_targets": batch["onset_targets"].to(device),
        "mask": batch["mask"].to(device),
    }


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    frame_pos_weight: float,
    onset_pos_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_loss = masked_weighted_bce(
        outputs["frame_logits"],
        batch["frame_targets"],
        batch["mask"],
        frame_pos_weight,
    )
    onset_loss = masked_weighted_bce(
        outputs["onset_logits"],
        batch["onset_targets"],
        batch["mask"],
        onset_pos_weight,
    )
    return frame_loss + onset_loss, frame_loss, onset_loss


def run_epoch(
    model: OnsetsAndFramesModel,
    loader: DataLoader,
    device: torch.device,
    frame_pos_weight: float,
    onset_pos_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "frame_loss": 0.0,
        "onset_loss": 0.0,
        "frame_precision": 0.0,
        "frame_recall": 0.0,
        "frame_f1": 0.0,
        "onset_precision": 0.0,
        "onset_recall": 0.0,
        "onset_f1": 0.0,
    }
    batch_count = 0

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(batch["features"])
            loss, frame_loss, onset_loss = compute_losses(
                outputs,
                batch,
                frame_pos_weight=frame_pos_weight,
                onset_pos_weight=onset_pos_weight,
            )
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        frame_metrics = compute_metrics(
            outputs["frame_logits"],
            batch["frame_targets"],
            batch["mask"],
        )
        onset_metrics = compute_metrics(
            outputs["onset_logits"],
            batch["onset_targets"],
            batch["mask"],
        )
        totals["loss"] += float(loss.detach().cpu())
        totals["frame_loss"] += float(frame_loss.detach().cpu())
        totals["onset_loss"] += float(onset_loss.detach().cpu())
        for name, value in frame_metrics.items():
            totals[f"frame_{name}"] += value
        for name, value in onset_metrics.items():
            totals[f"onset_{name}"] += value
        batch_count += 1

    if batch_count == 0:
        return totals
    return {name: value / batch_count for name, value in totals.items()}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train an onset-aware guitar transcriber.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--frame-pos-weight", type=float, default=5.0)
    parser.add_argument("--onset-pos-weight", type=float, default=20.0)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} loss={metrics['loss']:.4f} "
        f"frame_loss={metrics['frame_loss']:.4f} "
        f"onset_loss={metrics['onset_loss']:.4f} "
        f"frame_f1={metrics['frame_f1']:.3f} "
        f"onset_f1={metrics['onset_f1']:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    dataset = TranscriptionDataset(args.dataset, hop_length=args.hop_length)
    train_indices, val_indices = split_indices(
        len(dataset),
        val_split=args.val_split,
        seed=args.seed,
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_transcription_batch,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_transcription_batch,
    )

    device = torch.device(args.device)
    model = OnsetsAndFramesModel(hidden_size=args.hidden_size).to(device)
    first_batch = next(iter(train_loader))
    with torch.no_grad():
        model(move_batch(first_batch, device)["features"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            frame_pos_weight=args.frame_pos_weight,
            onset_pos_weight=args.onset_pos_weight,
            optimizer=optimizer,
        )
        print(format_metrics(f"epoch {epoch} train", train_metrics))
        if val_indices:
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                frame_pos_weight=args.frame_pos_weight,
                onset_pos_weight=args.onset_pos_weight,
            )
            print(format_metrics(f"epoch {epoch} val  ", val_metrics))

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_size": args.hidden_size,
                "hop_length": args.hop_length,
            },
        },
        args.checkpoint_out,
    )
    print(f"saved checkpoint to {args.checkpoint_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
