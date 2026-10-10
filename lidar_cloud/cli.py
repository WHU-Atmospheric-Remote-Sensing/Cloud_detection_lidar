"""Small command-line interface for prediction, calibration and verification."""

import argparse
import json
from pathlib import Path

from .evaluation import predict_vde, verify, write_layers
from .inference import CloudDetector
from .io import input_files, read_observation, write_prediction
from .statistics import compute_stats
from .weights import model_manifest, sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lidar cloud detection: inference, station MAD calibration and verification.")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Predict one H5 file or a directory, leaving inputs untouched.")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True, help="Separate output directory.")
    predict.add_argument("--station", help="Explicit station/instrument ID used to select MAD statistics.")
    predict.add_argument("--stats", help="Station statistics JSON; defaults to bundled CARLNET statistics.")
    predict.add_argument("--model", choices=["E1", "E2", "E3", "E4", "E5"], default="E3")
    predict.add_argument("--threshold", type=float, default=0.65)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--batch-size", type=int, default=1)
    predict.add_argument("--cadence-seconds", type=float, default=300)
    predict.add_argument("--gap-seconds", type=float)
    predict.add_argument("--profiles", help="Optional half-open profile slice, e.g. 0:10 or 0:1.")
    predict.add_argument("--no-plots", action="store_true", help="Skip the default time-height PNG overview.")
    predict.add_argument("--plot-height-km", type=float, default=10, help="PNG altitude/height ceiling in km; does not crop inference.")
    predict.add_argument("--plot-style", choices=["paper", "overview"], default="paper", help="Paper-style 1064 nm curtain, or diagnostic probability/mask panels.")
    plot = commands.add_parser("plot", help="Visualize saved prediction H5 files or verification JSON without inference.")
    plot.add_argument("--input", required=True, help="Prediction H5/directory, verification JSON or its directory.")
    plot.add_argument("--output", required=True, help="PNG output directory.")
    plot.add_argument("--observations", help="Optional original H5/directory for signal and manual-label panels.")
    plot.add_argument("--plot-height-km", type=float, default=10)
    plot.add_argument("--plot-style", choices=["paper", "overview"], default="paper")
    plot.add_argument("--cadence-seconds", type=float, default=300, help="Width of a single profile and time-gap display convention.")
    stats = commands.add_parser("compute-stats", help="Compute this instrument's Median/MAD from calibration H5 data.")
    stats.add_argument("--input", required=True)
    stats.add_argument("--station", required=True)
    stats.add_argument("--output", required=True)
    stats.add_argument("--instrument-id")
    stats.add_argument("--seed", type=int, default=20260923)
    stats.add_argument("--max-samples", type=int, default=1_000_000)
    check = commands.add_parser("verify", help="Recompute pixel and physical boundary metrics against manual labels.")
    check.add_argument("--data", required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--models", nargs="+", choices=["E1", "E2", "E3", "E4", "E5"], default=["E1", "E2", "E3", "E4", "E5"])
    check.add_argument("--stats")
    check.add_argument("--device", default="cpu")
    check.add_argument("--batch-size", type=int, default=1)
    check.add_argument("--protocol", choices=["paper", "deployment"], default="paper", help="paper preserves original file-order windows; deployment segments time gaps.")
    args = parser.parse_args(argv)
    try:
        if args.command == "plot":
            from .visualization import plot_saved_prediction, plot_verification
            source, destination = Path(args.input), Path(args.output)
            report_path = source / "verification.json" if source.is_dir() else source
            if report_path.suffix.lower() == ".json" and report_path.is_file():
                output = plot_verification(json.loads(report_path.read_text(encoding="utf-8")), destination / "metrics.png")
                print(output)
            else:
                files = input_files(source)
                if len({file.name for file in files}) != len(files):
                    raise ValueError("Input contains duplicate filenames; plot each directory separately.")
                for file in files:
                    output = plot_saved_prediction(file, destination / f"{file.stem}.png", args.observations,
                                                   args.plot_height_km, args.cadence_seconds, args.plot_style)
                    print(output)
        elif args.command == "compute-stats":
            compute_stats(args.input, args.station, args.output, args.max_samples, args.seed, args.instrument_id)
            print(f"Station statistics written to {args.output}")
        elif args.command == "verify":
            report = verify(args.data, args.output, args.models, args.stats, args.device, args.batch_size, args.protocol)
            print(f"Verification results: {args.output} ({report['dataset_role']}, {report['n_files']} files)")
            if report.get("passed") is False:
                print("FAILED: reference pixel metrics or physical boundary checks differ beyond tolerance.")
                return 1
        else:
            source, destination = Path(args.input).resolve(), Path(args.output).resolve()
            if destination == source or source.is_dir() and destination.is_relative_to(source):
                raise ValueError("Output must be outside the input directory.")
            files = input_files(source)
            if len({f.name for f in files}) != len(files):
                raise ValueError("Input contains duplicate filenames; run each directory separately.")
            detector = None if args.model == "E5" else CloudDetector(args.model, args.stats, args.device, args.threshold, args.batch_size)
            for file in files:
                obs = read_observation(file, args.station, require_vde=args.model == "E5")
                if args.profiles:
                    parts = args.profiles.split(":")
                    if len(parts) != 2:
                        raise ValueError("--profiles must be start:stop, e.g. 0:10.")
                    start, stop = (int(part) if part else None for part in parts)
                    selection = slice(start, stop)
                    obs.signals = {name: values[selection] for name, values in obs.signals.items()}
                    obs.time_utc = obs.time_utc[selection]
                    obs.truth = None if obs.truth is None else obs.truth[selection]
                    obs.vde_signals = None if obs.vde_signals is None else tuple(v[selection] for v in obs.vde_signals)
                result = predict_vde(obs) if args.model == "E5" else detector.predict(obs.signals, obs.height_km, obs.time_utc, obs.station_id, args.cadence_seconds, args.gap_seconds)
                metadata = {"model": args.model, "source_file": file.name, "source_sha256": sha256_file(file),
                            "threshold": args.threshold, "notes": json.dumps(result.notes)}
                metadata["profile_selection"] = args.profiles or "all"
                if args.model != "E5":
                    metadata["weights_sha256"] = model_manifest()[args.model]["sha256"]
                else:
                    from .vde import mask_attributes
                    metadata.pop("threshold")
                    metadata.update(mask_attributes())
                output = destination / f"{file.stem}_{args.model}.h5"
                write_prediction(output, obs, result, metadata)
                write_layers(output.with_suffix(".csv"), result, obs)
                print(output)
                if not args.no_plots:
                    from .visualization import plot_prediction
                    figure = plot_prediction(obs, result, output.with_suffix(".png"), metadata,
                                             args.plot_height_km, args.cadence_seconds, args.gap_seconds, args.plot_style)
                    print(figure)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0
