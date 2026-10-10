"""Original 2048 -> 1056 range compression and logit interpolation."""
import numpy as np
import torch

def compress_z_axis(x, height_km, i0=800, i1=1600, factor_mid=4, factor_top=8):
    is_torch = isinstance(x, torch.Tensor)
    device = x.device if is_torch else None
    if is_torch and x.dim() == 4:
        x_list = []
        for i in range(x.shape[0]):
            x_i, h_new, z_low_idx, comp_info = compress_z_axis(x[i], height_km, i0, i1, factor_mid, factor_top)
            x_list.append(x_i.unsqueeze(0))
        x_new = torch.cat(x_list, dim=0)
        return (x_new, h_new, z_low_idx, comp_info)
    if is_torch:
        x_np = x.cpu().numpy()
    else:
        x_np = x
    h_np = height_km if isinstance(height_km, np.ndarray) else height_km.cpu().numpy()
    if x_np.ndim == 3:
        T, Z_raw, C = x_np.shape
    else:
        raise ValueError(f'Expected x with shape (T, Z_raw, C), got {x_np.shape}')
    x_low = x_np[:, :i0, :]
    h_low = h_np[:i0]
    x_mid_full = x_np[:, i0:i1, :]
    h_mid_full = h_np[i0:i1]
    Z_mid_full = x_mid_full.shape[1]
    Z_mid_new = Z_mid_full // factor_mid
    if Z_mid_new > 0:
        x_mid_valid = x_mid_full[:, :Z_mid_new * factor_mid, :]
        h_mid_valid = h_mid_full[:Z_mid_new * factor_mid]
        x_mid = x_mid_valid.reshape(T, Z_mid_new, factor_mid, C).mean(axis=2)
        h_mid = h_mid_valid.reshape(Z_mid_new, factor_mid).mean(axis=1)
    else:
        x_mid = np.zeros((T, 0, C), dtype=x_np.dtype)
        h_mid = np.zeros((0,), dtype=h_np.dtype)
    x_top_full = x_np[:, i1:, :]
    h_top_full = h_np[i1:]
    Z_top_full = x_top_full.shape[1]
    Z_top_new = Z_top_full // factor_top
    if Z_top_new > 0:
        x_top_valid = x_top_full[:, :Z_top_new * factor_top, :]
        h_top_valid = h_top_full[:Z_top_new * factor_top]
        x_top = x_top_valid.reshape(T, Z_top_new, factor_top, C).mean(axis=2)
        h_top = h_top_valid.reshape(Z_top_new, factor_top).mean(axis=1)
    else:
        x_top = np.zeros((T, 0, C), dtype=x_np.dtype)
        h_top = np.zeros((0,), dtype=h_np.dtype)
    x_new_np = np.concatenate([x_low, x_mid, x_top], axis=1)
    h_new_np = np.concatenate([h_low, h_mid, h_top], axis=0)
    z_low_idx = x_low.shape[1]
    z_mid_start = z_low_idx
    z_mid_end = z_mid_start + x_mid.shape[1]
    z_top_start = z_mid_end
    compression_info = {'Z_raw': Z_raw, 'Z_new': x_new_np.shape[1], 'i0': i0, 'i1': i1, 'factor_mid': factor_mid, 'factor_top': factor_top, 'z_low_idx': z_low_idx, 'z_mid_range': (z_mid_start, z_mid_end), 'z_top_range': (z_top_start, x_new_np.shape[1]), 'raw_ranges': {'low': (0, i0), 'mid': (i0, i0 + Z_mid_new * factor_mid), 'top': (i1, i1 + Z_top_new * factor_top)}}
    if is_torch:
        x_new = torch.from_numpy(x_new_np).to(device)
        h_new = torch.from_numpy(h_new_np).to(device)
    else:
        x_new = x_new_np
        h_new = h_new_np
    return (x_new, h_new, z_low_idx, compression_info)

def decompress_z_predictions(pred_compressed, compression_info, interpolate=True):
    is_torch = isinstance(pred_compressed, torch.Tensor)
    has_batch = pred_compressed.dim() == 4
    Z_raw = compression_info['Z_raw']
    z_ranges = compression_info['raw_ranges']
    factor_mid = compression_info['factor_mid']
    factor_top = compression_info['factor_top']
    z_low_idx = compression_info['z_low_idx']
    z_mid_range = compression_info['z_mid_range']
    z_top_range = compression_info['z_top_range']
    if is_torch:
        device = pred_compressed.device
        pred_np = pred_compressed.cpu().numpy()
    else:
        pred_np = pred_compressed
    if has_batch:
        B = pred_np.shape[0]
        pred_list = []
        for i in range(B):
            pred_i = decompress_z_predictions(pred_compressed[i] if is_torch else pred_np[i], compression_info, interpolate)
            pred_list.append(pred_i[np.newaxis] if not is_torch else pred_i.unsqueeze(0))
        if is_torch:
            return torch.cat(pred_list, dim=0)
        else:
            return np.concatenate(pred_list, axis=0)
    T, Z_new, C = pred_np.shape
    pred_low = pred_np[:, :z_low_idx, :]
    pred_mid = pred_np[:, z_mid_range[0]:z_mid_range[1], :]
    pred_top = pred_np[:, z_top_range[0]:z_top_range[1], :]
    pred_raw_np = np.zeros((T, Z_raw, C), dtype=pred_np.dtype)
    pred_raw_np[:, :z_ranges['low'][1], :] = pred_low
    if pred_mid.shape[1] > 0:
        if interpolate:
            from scipy.interpolate import interp1d
            z_mid_compressed = np.arange(pred_mid.shape[1])
            z_mid_target = np.linspace(0, pred_mid.shape[1] - 1, z_ranges['mid'][1] - z_ranges['mid'][0])
            for t in range(T):
                for c in range(C):
                    f = interp1d(z_mid_compressed, pred_mid[t, :, c], kind='linear', fill_value='extrapolate')
                    pred_raw_np[t, z_ranges['mid'][0]:z_ranges['mid'][1], c] = f(z_mid_target)
        else:
            pred_mid_repeat = np.repeat(pred_mid, factor_mid, axis=1)
            pred_raw_np[:, z_ranges['mid'][0]:z_ranges['mid'][1], :] = pred_mid_repeat
    if pred_top.shape[1] > 0:
        if interpolate:
            from scipy.interpolate import interp1d
            z_top_compressed = np.arange(pred_top.shape[1])
            z_top_target = np.linspace(0, pred_top.shape[1] - 1, z_ranges['top'][1] - z_ranges['top'][0])
            for t in range(T):
                for c in range(C):
                    f = interp1d(z_top_compressed, pred_top[t, :, c], kind='linear', fill_value='extrapolate')
                    pred_raw_np[t, z_ranges['top'][0]:z_ranges['top'][1], c] = f(z_top_target)
        else:
            pred_top_repeat = np.repeat(pred_top, factor_top, axis=1)
            pred_raw_np[:, z_ranges['top'][0]:z_ranges['top'][1], :] = pred_top_repeat
    if is_torch:
        return torch.from_numpy(pred_raw_np).to(device)
    else:
        return pred_raw_np
