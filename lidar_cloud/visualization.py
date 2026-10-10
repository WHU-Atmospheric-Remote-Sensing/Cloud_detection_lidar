"""Headless PNG figures from predictions and measured verification results."""

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .inference import Prediction
from .io import Observation, input_files, read_observation
from .preprocessing import validate_coordinates


def _figure(size):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _time_cells(times, cadence_seconds, gap_seconds):
    """Separate meshes leave missing intervals blank, including single profiles."""
    import matplotlib.dates as dates
    x = dates.date2num([datetime.fromtimestamp(float(t), timezone.utc) for t in times])
    breaks = np.flatnonzero(np.diff(times) > gap_seconds) + 1
    segments = []
    for indices in np.split(np.arange(len(times)), breaks):
        centers = x[indices]
        if len(indices) == 1:
            half = cadence_seconds / (2 * 86400)
            edges = np.array([centers[0] - half, centers[0] + half])
        else:
            midpoints = (centers[1:] + centers[:-1]) / 2
            edges = np.r_[centers[0] - (midpoints[0] - centers[0]), midpoints,
                          centers[-1] + (centers[-1] - midpoints[-1])]
        segments.append((indices, edges))
    return x, segments


def _plot_overview(observation, result, output, metadata=None, max_height_km=10,
                    cadence_seconds=300, gap_seconds=None):
    """Write a time-height overview; optional input signal and labels add panels."""
    import matplotlib.dates as dates
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from .evaluation import extract_layers

    height, times = validate_coordinates(observation.height_km, observation.time_utc)
    shape = (len(times), len(height))
    if any(np.shape(v) != shape for v in (result.probability, result.mask, result.valid)):
        raise ValueError("Prediction arrays must match the time/height coordinates.")
    if max_height_km is not None and (not np.isfinite(max_height_km) or max_height_km <= height[0]):
        raise ValueError("Plot height ceiling must be finite and above the first measured height.")
    gap = float(gap_seconds) if gap_seconds is not None else 1.5 * cadence_seconds
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0 or not np.isfinite(gap) or gap <= 0:
        raise ValueError("Plot cadence and gap must be finite and positive.")
    x, segments = _time_cells(times, cadence_seconds, gap)
    selected = height < max_height_km if max_height_km is not None else np.ones(len(height), dtype=bool)
    # The layer convention treats the final positive bin's upper edge as its top.
    z_edges = np.r_[height, height[-1] + np.median(np.diff(height))]
    z_edges = z_edges[:int(selected.sum()) + 1]
    discrete = ListedColormap(["#edf2f7", "#1768ac", "#b8bec7"])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], discrete.N)
    mask = np.where(result.valid, result.mask == 1, 2)
    panels = []
    if "1064" in observation.signals:
        raw = np.asarray(observation.signals["1064"], dtype=float)
        displayed = np.log10(1 + np.maximum(raw, 0) * height[None, :] ** 2)
        displayed[~result.valid] = np.nan
        finite = displayed[:, selected][np.isfinite(displayed[:, selected])]
        limits = np.percentile(finite, [2, 98]) if finite.size else [0, 1]
        if limits[1] <= limits[0]:
            limits = [limits[0], limits[0] + 1]
        panels.append(("1064 nm signal (display transform only)", displayed, "magma", None,
                       limits, "log10(1 + positive signal × range²)"))
    binary = (metadata or {}).get("model") == "E5"
    panels.append(("VDE binary decision" if binary else "Cloud probability", result.probability,
                   "viridis", None, [0, 1], "Decision" if binary else "Probability"))
    panels.append(("Detected mask and layer boundaries", mask, discrete, norm, None, None))
    if observation.truth is not None:
        truth = np.where(result.valid, observation.truth > 0, 2)
        panels.append(("Manual cloud annotation", truth, discrete, norm, None, None))
    figure = _figure((12, 2.5 * len(panels) + .5))
    axes = figure.subplots(len(panels), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    mask_axis = None
    for axis, (title, values, cmap, color_norm, limits, label) in zip(axes, panels):
        kwargs = {"cmap": cmap, "norm": color_norm} if color_norm is not None else {"cmap": cmap, "vmin": limits[0], "vmax": limits[1]}
        for indices, edges in segments:
            mesh = axis.pcolormesh(edges, z_edges, np.ma.masked_invalid(values[indices][:, selected].T),
                                   shading="flat", rasterized=True, **kwargs)
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_ylabel("Height AGL (km)")
        axis.set_ylim(height[0], min(z_edges[-1], max_height_km) if max_height_km is not None else z_edges[-1])
        if color_norm is not None:
            colorbar = figure.colorbar(mesh, ax=axis, ticks=[0, 1, 2], pad=.015, fraction=.025)
            colorbar.ax.set_yticklabels(["Clear", "Cloud", "Invalid"])
            if title.startswith("Detected"):
                mask_axis = axis
        else:
            figure.colorbar(mesh, ax=axis, label=label, pad=.015, fraction=.025)
    boundary_x, bases, tops = [], [], []
    for index, row in enumerate(result.mask):
        for layer in extract_layers((row == 1) & result.valid[index], height):
            boundary_x.append(x[index])
            bases.append(layer["base_m"] / 1000)
            tops.append(layer["top_m"] / 1000)
    mask_axis.scatter(boundary_x, bases, s=5, color="#d85c00", marker=".", zorder=3)
    mask_axis.scatter(boundary_x, tops, s=5, color="#891aa9", marker=".", zorder=3)
    mask_axis.legend(handles=[Line2D([], [], marker=".", ls="", color="#d85c00", label="Cloud base"),
                              Line2D([], [], marker=".", ls="", color="#891aa9", label="Detectable cloud top")],
                     loc="lower right", bbox_to_anchor=(1, 1.01), frameon=False, fontsize=8, ncols=2)
    locator = dates.AutoDateLocator(minticks=3, maxticks=10, tz=timezone.utc)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(dates.ConciseDateFormatter(locator, tz=timezone.utc))
    axes[-1].set_xlabel("Time (UTC); gaps are blank")
    metadata = metadata or {}
    threshold = f" · threshold {metadata['threshold']:g}" if "threshold" in metadata else ""
    figure.suptitle(f"{observation.station_id} · {metadata.get('model', 'Cloud detection')}{threshold}", fontsize=12)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, metadata={"PlotStyle": "overview",
                   "HeightCeilingKm": str(axes[0].get_ylim()[1]), "HeightReference": "AGL"})
    return output


def plot_prediction(observation, result, output, metadata=None, max_height_km=10,
                    cadence_seconds=300, gap_seconds=None, style="paper"):
    """Default: manuscript-style 1064 nm curtain with outlined cloud masks."""
    import matplotlib
    if style not in {"paper", "overview"}:
        raise ValueError("Plot style must be paper or overview.")
    with matplotlib.rc_context({"font.family": "Arial", "axes.unicode_minus": False,
                                "pdf.fonttype": 42, "ps.fonttype": 42}):
        renderer = _plot_paper if style == "paper" else _plot_overview
        return renderer(observation, result, output, metadata, max_height_km, cadence_seconds, gap_seconds)


def _draw_mask_contour(axis, x, height, mask, valid, edges, linestyle, bin_depth_km=None):
    """Draw real mask geometry separately for each time segment."""
    import matplotlib.patheffects as effects
    from matplotlib.patches import Rectangle
    halo = [effects.Stroke(linewidth=4, foreground="white", alpha=.95), effects.Normal()]
    if not np.any(mask & valid):
        return
    if len(x) == 1 or len(height) == 1:
        dz = bin_depth_km if bin_depth_km is not None else float(np.median(np.diff(height)))
        for index, row in enumerate(mask & valid):
            cloud = np.flatnonzero(row)
            if not len(cloud):
                continue
            for group in np.split(cloud, np.flatnonzero(np.diff(cloud) > 1) + 1):
                base, top = height[group[0]], height[group[-1]] + dz
                patch = Rectangle((edges[index], base), edges[index + 1] - edges[index], top - base,
                                  fill=False, edgecolor="black", linewidth=2, linestyle=linestyle, zorder=6)
                patch.set_path_effects(halo)
                axis.add_patch(patch)
        return
    # Padding here only closes the displayed contour at observation edges;
    # it never changes inference, probabilities or the saved mask.
    field = np.where(valid, mask.astype(float), np.nan).T
    field = np.pad(field, ((1, 1), (1, 1)), constant_values=0)
    dz = np.median(np.diff(height))
    z = np.r_[height[0] - dz, height, height[-1] + dz]
    t = np.r_[2 * edges[0] - x[0], x, 2 * edges[-1] - x[-1]]
    contour = axis.contour(t, z, np.ma.masked_invalid(field), levels=[.5], colors="black",
                           linestyles=linestyle, linewidths=2, zorder=6)
    if hasattr(contour, "set_path_effects"):
        contour.set_path_effects(halo)
    else:
        for collection in contour.collections:  # Matplotlib 3.7 compatibility.
            collection.set_path_effects(halo)


def _plot_paper(observation, result, output, metadata, max_height_km, cadence_seconds, gap_seconds):
    import matplotlib.dates as dates
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap
    from .io import station_elevation_m
    height, times = validate_coordinates(observation.height_km, observation.time_utc)
    shape = (len(times), len(height))
    if any(np.shape(array) != shape for array in (result.probability, result.mask, result.valid)):
        raise ValueError("Prediction arrays must match the time/height coordinates.")
    gap = float(gap_seconds) if gap_seconds is not None else 1.5 * cadence_seconds
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0 or not np.isfinite(gap) or gap <= 0:
        raise ValueError("Plot cadence and gap must be finite and positive.")
    elevation = station_elevation_m(observation.station_id, observation.attrs)
    altitude = height + (elevation / 1000 if elevation is not None else 0)
    if max_height_km is not None and (not np.isfinite(max_height_km) or max_height_km <= altitude[0]):
        raise ValueError("Plot height ceiling must be finite and above the first measured altitude.")
    selected = altitude < max_height_km if max_height_km is not None else np.ones(len(height), dtype=bool)
    z = altitude[selected]
    z_edges = np.r_[altitude, altitude[-1] + np.median(np.diff(altitude))][:len(z) + 1]
    x, segments = _time_cells(times, cadence_seconds, gap)
    figure = _figure((12, 5.4))
    axis = figure.subplots()
    raw = observation.signals.get("1064")
    if raw is not None:
        if np.shape(raw) != shape:
            raise ValueError("1064 nm signal must match prediction coordinates.")
        # Exactly the paper background transform; ground-relative range in km.
        background = np.log1p(np.maximum(np.asarray(raw, dtype=float), 0) * height[None, :] ** 2)
        background[~result.valid] = np.nan
        scale_region = selected & (height >= .5)
        finite = background[:, scale_region][np.isfinite(background[:, scale_region])]
        if not len(finite):
            finite = background[:, selected][np.isfinite(background[:, selected])]
        limits = np.percentile(finite, [2, 98]) if len(finite) else np.array([0, 1])
        if limits[1] <= limits[0]:
            limits[1] = limits[0] + 1
        for indices, edges in segments:
            mesh = axis.pcolormesh(edges, z_edges, np.ma.masked_invalid(background[indices][:, selected].T),
                                   cmap="turbo", vmin=limits[0], vmax=limits[1], shading="flat", rasterized=True)
        figure.colorbar(mesh, ax=axis, label=r"$\log(P\cdot R^2)$", pad=.02, fraction=.03)
    else:
        # Saved H5 alone still gives an honest mask view; no synthetic signal.
        values = np.where(result.valid, result.mask == 1, 2)
        for indices, edges in segments:
            mesh = axis.pcolormesh(edges, z_edges, values[indices][:, selected].T, shading="flat",
                                   cmap=ListedColormap(["white", "#2171b5", "#b8bec7"]), vmin=-.5, vmax=2.5)
        colorbar = figure.colorbar(mesh, ax=axis, ticks=[0, 1, 2], pad=.02, fraction=.03)
        colorbar.ax.set_yticklabels(["Clear", "Cloud", "Invalid"])
    # Gray identifies unsupported/nonfinite observations; time gaps stay blank.
    for indices, edges in segments:
        invalid = np.ma.masked_where(result.valid[indices][:, selected].T,
                                    np.ones((len(z), len(indices))))
        axis.pcolormesh(edges, z_edges, invalid, cmap=ListedColormap(["#b8bec7"]),
                        vmin=0, vmax=1, shading="flat", rasterized=True, zorder=3)
        if observation.truth is not None:
            _draw_mask_contour(axis, x[indices], z, observation.truth[indices][:, selected] > 0,
                               result.valid[indices][:, selected], edges, ":", float(np.median(np.diff(altitude))))
        _draw_mask_contour(axis, x[indices], z, result.mask[indices][:, selected] == 1,
                           result.valid[indices][:, selected], edges, "-", float(np.median(np.diff(altitude))))
    metadata = metadata or {}
    name = metadata.get("model", "Cloud detection")
    legend = [Line2D([], [], color="black", linewidth=2, label=f"{name} cloud mask")]
    if observation.truth is not None:
        legend.append(Line2D([], [], color="black", ls=":", linewidth=2, label="Manual annotation"))
    axis.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, -.31), ncols=2,
                frameon=True, edgecolor="#777777", handlelength=3)
    date = datetime.fromtimestamp(float(times[0]), timezone.utc).strftime("%Y-%m-%d")
    axis.set_title(f"Station {observation.station_id} · {date} · {name}", loc="left", fontsize=15, fontweight="bold")
    axis.set_ylabel("Altitude\n(km a.s.l.)" if elevation is not None else "Height\n(km a.g.l.)",
                    fontsize=13, fontweight="bold")
    axis.set_ylim(min(0, altitude[0]), min(z_edges[-1], max_height_km) if max_height_km is not None else z_edges[-1])
    axis.set_xlim(segments[0][1][0], segments[-1][1][-1])
    axis.set_xlabel("Time (UTC)", fontsize=13, fontweight="bold")
    locator = dates.AutoDateLocator(minticks=3, maxticks=10, tz=timezone.utc)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator, tz=timezone.utc))
    axis.tick_params(labelsize=11, width=1.2, length=6)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white", metadata={"PlotStyle": "paper",
                   "HeightCeilingKm": str(axis.get_ylim()[1]),
                   "HeightReference": "ASL" if elevation is not None else "AGL"})
    return output


def plot_saved_prediction(input_path, output, observations=None, max_height_km=10, cadence_seconds=300, style="paper"):
    """Plot a prediction H5 without rerunning a model, optionally with its source."""
    with h5py.File(input_path, "r") as h:
        metadata = dict(h.attrs)
        height, times = h["coordinates/height_km"][:], h["coordinates/time_utc"][:]
        result = Prediction(h["prediction/cloud_probability"][:], h["prediction/cloud_mask"][:],
                            h["prediction/valid_observation"][:].astype(bool))
    obs = Observation({}, height, times, str(metadata.get("station_id", "Unknown")), metadata)
    if observations is not None:
        name = str(metadata.get("source_file", ""))
        matches = [file for file in input_files(observations) if file.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one source observation named {name!r}.")
        original = read_observation(matches[0])
        indices = np.searchsorted(original.time_utc, times)
        if np.any(indices >= len(original.time_utc)) or not np.allclose(original.time_utc[indices], times, rtol=0, atol=.001):
            raise ValueError("Prediction timestamps do not match source observations.")
        if len(original.height_km) != len(height) or not np.allclose(original.height_km, height, rtol=0, atol=1e-6):
            raise ValueError("Prediction heights do not match source observations.")
        obs.signals = {name: array[indices] for name, array in original.signals.items()}
        obs.truth = original.truth[indices] if original.truth is not None else None
    return plot_prediction(obs, result, output, metadata, max_height_km, cadence_seconds, style=style)


def plot_verification(report, output):
    """Plot measured pixel and boundary metrics directly from a verification JSON."""
    figure = _figure((14, 4.2))
    axes = figure.subplots(1, 3)
    pixel = report["pixel_metrics"]
    names = [row["method"] for row in pixel]
    x = np.arange(len(names))
    colors = ["#1768ac", "#479c6f", "#d48531", "#8056a8"]
    for offset, (key, color) in enumerate(zip(["iou", "f1", "precision", "recall"], colors)):
        axes[0].bar(x + (offset - 1.5) * .18, [row[key] for row in pixel], .18, label=key.title(), color=color)
    axes[0].set_title("Pixel metrics · global micro counts")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, ncols=4, loc="upper center", bbox_to_anchor=(.5, -.13))
    boundary = {row["method"]: row for row in report["boundary_metrics"]}
    for offset, (key, label, color) in enumerate([("base_mae_m", "Cloud base", colors[0]),
                                               ("top_mae_m", "Detectable top", colors[2])]):
        axes[1].bar(x + (offset - .5) * .32,
                    [boundary[name][key] if boundary[name][key] is not None else np.nan for name in names],
                    .32, label=label, color=color)
    axes[1].set_title("Matched layer boundary errors")
    axes[1].set_ylabel("Mean absolute error (m)")
    axes[1].legend(fontsize=8)
    axes[2].bar(x, [boundary[name]["detection_rate"] for name in names], .6, color=colors[1])
    axes[2].set_title("Manual layer detection rate")
    axes[2].set_ylim(0, 1.05)
    for axis in axes:
        axis.set_xticks(x, names)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    role = "Full test" if report.get("dataset_role") == "full_test" else "Example/local subset"
    figure.suptitle(f"{role} · {report['n_files']} files · {report.get('protocol', 'paper')} protocol", fontsize=12)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    return output
