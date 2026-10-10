"""Run 10 profiles; adapt write_minimal_h5 for your own observations."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from lidar_cloud import CloudDetector, load_backbone, load_model
from lidar_cloud.io import CHANNELS, read_observation, write_prediction
from lidar_cloud.evaluation import write_layers
from lidar_cloud.visualization import plot_prediction


def write_minimal_h5(path, signals, height_km, time_utc, station_id):
    """signals contains five (T,Z) raw observation-level arrays."""
    with h5py.File(path, "w") as h:
        h.attrs["station_id"] = station_id
        h.create_dataset("coordinates/time_utc", data=time_utc)
        h.create_dataset("coordinates/height_km", data=height_km)
        for channel in CHANNELS:
            group = "depolarization" if channel.startswith("vdr_") else "preprocessed_signals"
            h.create_dataset(f"{group}/{channel}", data=signals[channel], compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(next((Path(__file__).resolve().parents[1] / "data/examples").glob("58847*.h5"))))
    parser.add_argument("--profiles", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/python_example")
    args = parser.parse_args()
    obs = read_observation(args.input)
    obs.signals = {ch: values[:args.profiles] for ch, values in obs.signals.items()}
    obs.time_utc = obs.time_utc[:args.profiles]
    obs.truth = obs.truth[:args.profiles] if obs.truth is not None else None
    result = CloudDetector(device=args.device).predict(
        obs.signals, obs.height_km, obs.time_utc, obs.station_id,
    )
    output = Path(args.output) / f"{Path(args.input).stem}_E3.h5"
    metadata = {"model": "E3", "threshold": .65, "source_file": Path(args.input).name}
    write_prediction(output, obs, result, metadata)
    write_layers(output.with_suffix(".csv"), result, obs)
    plot_prediction(obs, result, output.with_suffix(".png"), metadata)
    print(f"Probability shape: {result.probability.shape}; cloud pixels: {np.count_nonzero(result.mask == 1)}")
    print(f"H5, CSV and PNG written to {output.parent}")
    # Other-algorithm initialization (parameter loading only; no training pipeline):
    # pretrained = load_model("R1")  # original reconstruction and segmentation heads
    # backbone = load_backbone("R2")  # attach your own task head
    # features = backbone(normalized_compressed_tensor)  # (B,T,1056,5) -> (B,48,T,1056)
