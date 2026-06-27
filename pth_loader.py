"""Inspect the top-level and model keys in a PyTorch checkpoint."""

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to a .pth/.pt checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        print(f"Checkpoint type: {type(checkpoint).__name__}")
        return

    print("Checkpoint keys:", sorted(checkpoint))
    model = checkpoint.get("model")
    if isinstance(model, dict):
        print("Model keys:")
        for key in model:
            print(key)


if __name__ == "__main__":
    main()
