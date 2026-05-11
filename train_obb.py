from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolo11m-obb.pt"
DEFAULT_DATA = ROOT / "bottle_obb.yaml"


def parse_args():
    parser = ArgumentParser(description="Train YOLO11m-OBB on the bottle dataset.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to the OBB pretrained weights.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to the dataset yaml.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="Use 0 for GPU, cpu for CPU, or omit for auto.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default=None, help="Run name. Defaults to a timestamped name.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "obb"))
    parser.add_argument("--resume", action="store_true", help="Resume the latest interrupted training run.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing run directory.")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    run_name = args.name or f"bottle_yolo11m_obb_{datetime.now():%Y%m%d_%H%M%S}"

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        task="obb",
        resume=args.resume,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
