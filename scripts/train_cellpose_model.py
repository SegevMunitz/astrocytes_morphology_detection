"""Train one checkpoint-preserving Cellpose 3 model with held-out validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from cellpose import io, models, train


def parse_args() -> argparse.Namespace:
    """Parse reproducible optimization, data, and output parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--pretrained-model", default="cyto2_cp3")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--chan", type=int, default=1)
    parser.add_argument("--chan2", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Fine-tune Cellpose and persist per-epoch train/validation loss history."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Cellpose sweep")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    np.random.seed(42)
    torch.manual_seed(42)
    io.logger_setup(cp_path=str(args.output_dir), logfile_name="cellpose.log")
    loaded = io.load_train_test_data(
        str(args.train_dir),
        str(args.validation_dir),
        mask_filter="_seg.npy",
    )
    images, labels, image_names, validation_images, validation_labels, validation_names = loaded
    if not images or not validation_images:
        raise ValueError("Both training and validation sets must contain labeled images")

    model = models.CellposeModel(gpu=True, model_type=args.pretrained_model)
    model_path, train_losses, validation_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        train_files=image_names,
        test_data=validation_images,
        test_labels=validation_labels,
        test_files=validation_names,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.epochs,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        SGD=True,
        channels=[args.chan, args.chan2],
        save_path=args.output_dir,
        save_every=args.save_every,
        save_each=True,
        model_name=args.model_name,
    )

    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_loss", "validation_loss"))
        writer.writerows(
            (epoch, float(train_loss), float(validation_loss))
            for epoch, (train_loss, validation_loss) in enumerate(
                zip(train_losses, validation_losses, strict=True)
            )
        )
    metadata = {
        "model_path": str(model_path),
        "pretrained_model": args.pretrained_model,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "channels": [args.chan, args.chan2],
        "training_images": [str(path) for path in image_names],
        "validation_images": [str(path) for path in validation_names],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Cellpose model saved to {model_path}")


if __name__ == "__main__":
    main()
