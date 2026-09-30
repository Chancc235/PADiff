"""Add reward-to-go to processed episodes for PADiff training."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=("PP4a", "LBF", "overcooked"))
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parents[1] / "data"
    input_path = data_dir / f"{args.env}_episodes_datas.pt"
    output_path = data_dir / f"{args.env}_data.pt"

    episodes = torch.load(input_path)
    for episode in episodes:
        rewards = episode["reward"].reshape(-1)
        episode["rtg"] = torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=0), dims=(0,))
    torch.save(episodes, output_path)
    print(f"Saved {len(episodes)} episodes to {output_path}")


if __name__ == "__main__":
    main()
