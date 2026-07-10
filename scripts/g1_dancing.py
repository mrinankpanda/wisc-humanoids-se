import argparse
from simulation import Simulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", required=True, help="Path to the motion .npz file")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless (no popup window)")
    args = parser.parse_args()

    sim = Simulator(npz_path=args.npz_path, show_viewer=not args.no_viewer)
    sim.run()