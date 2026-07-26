"""Deterministic fixtures for signal-processing and forensic unit tests."""

import cv2
import numpy as np


def sinusoidal_grating(
    size=256,
    period_pixels=12.0,
    angle_degrees=0.0,
    amplitude=30.0,
    mean=127.0,
    noise_standard_deviation=0.0,
    seed=1,
):
    yy, xx = np.indices((size, size), dtype=np.float32)
    angle = np.deg2rad(angle_degrees)
    coordinate = xx * np.cos(angle) + yy * np.sin(angle)
    image = mean + amplitude * np.sin(
        2.0 * np.pi * coordinate / period_pixels
    )
    if noise_standard_deviation > 0.0:
        random = np.random.default_rng(seed)
        image += random.normal(0.0, noise_standard_deviation, image.shape)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def checkerboard(size=256, cell_size=8, low=45, high=210):
    yy, xx = np.indices((size, size))
    pattern = ((xx // cell_size + yy // cell_size) % 2).astype(np.uint8)
    return np.where(pattern, high, low).astype(np.uint8)


def screen_lattice(size=256, period_x=9.0, period_y=11.0, amplitude=25.0):
    yy, xx = np.indices((size, size), dtype=np.float32)
    image = (
        127.0
        + amplitude * np.sin(2.0 * np.pi * xx / period_x)
        + amplitude * np.sin(2.0 * np.pi * yy / period_y)
    )
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def jpeg_recompress(image, quality_sequence):
    result = np.asarray(image, dtype=np.uint8)
    for quality in quality_sequence:
        success, encoded = cv2.imencode(
            ".jpg",
            result,
            [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
        )
        if not success:
            raise RuntimeError("OpenCV could not encode synthetic JPEG")
        result = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return result


def add_eight_pixel_block_steps(image, strength=18.0):
    result = np.asarray(image, dtype=np.float32).copy()
    for boundary in range(8, result.shape[1], 8):
        result[:, boundary:] += strength * (
            1.0 if (boundary // 8) % 2 else -1.0
        )
    for boundary in range(8, result.shape[0], 8):
        result[boundary:, :] += 0.5 * strength * (
            1.0 if (boundary // 8) % 2 else -1.0
        )
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def sharpen(image, amount=1.0, sigma=1.2):
    source = np.asarray(image, dtype=np.float32)
    blurred = cv2.GaussianBlur(source, (0, 0), sigma)
    return np.clip(source + amount * (source - blurred), 0.0, 255.0).astype(
        np.uint8
    )


def resize_roundtrip(image, scale=0.55):
    source = np.asarray(image)
    height, width = source.shape[:2]
    small = cv2.resize(
        source,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def add_periodic_residual(image, period=7.0, amplitude=8.0):
    source = np.asarray(image, dtype=np.float32)
    yy, xx = np.indices(source.shape[:2], dtype=np.float32)
    residual = amplitude * np.sin(2.0 * np.pi * (xx + 0.35 * yy) / period)
    if source.ndim == 3:
        residual = residual[:, :, None]
    return np.clip(source + residual, 0.0, 255.0).astype(np.uint8)


def clip_colors(image, lower=30, upper=220):
    return np.clip(np.asarray(image), lower, upper).astype(np.uint8)


def chromaticity_shift(bgr_image, blue_gain=1.0, green_gain=1.0, red_gain=1.0):
    source = np.asarray(bgr_image, dtype=np.float32)
    gains = np.asarray([blue_gain, green_gain, red_gain], dtype=np.float32)
    return np.clip(source * gains, 0.0, 255.0).astype(np.uint8)


def temporal_sinusoid(timestamps_s, frequency_hz, amplitude=1.0, phase=0.0):
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    return amplitude * np.sin(
        2.0 * np.pi * frequency_hz * timestamps + phase
    )


def irregular_timestamps(frame_count=90, fps=30.0, gap_every=10):
    intervals = np.full(frame_count - 1, 1.0 / fps, dtype=np.float64)
    intervals[gap_every - 1 :: gap_every] *= 3.0
    return np.concatenate(([0.0], np.cumsum(intervals)))


def planar_motion_sequence(image, frame_count=8, translation_pixels=2.0):
    source = np.asarray(image)
    height, width = source.shape[:2]
    frames = []
    for index in range(frame_count):
        transform = np.float32(
            [[1.0, 0.0, translation_pixels * index], [0.0, 1.0, 0.4 * index]]
        )
        frames.append(
            cv2.warpAffine(
                source,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return frames


def nonplanar_motion_sequence(image, frame_count=8, amplitude_pixels=3.0):
    frames = planar_motion_sequence(image, frame_count, 1.0)
    height, width = np.asarray(image).shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    output = []
    for index, frame in enumerate(frames):
        displacement = amplitude_pixels * np.sin(
            2.0 * np.pi * xx / max(width, 1) + 0.5 * index
        )
        map_x = xx + displacement * (yy / max(height - 1, 1))
        output.append(
            cv2.remap(
                frame,
                map_x,
                yy,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return output


def pulse_rgb_signals(timestamps_s, pulse_hz=1.2, amplitude=0.01):
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    pulse = np.sin(2.0 * np.pi * pulse_hz * timestamps)
    return np.column_stack(
        (
            0.62 + 0.6 * amplitude * pulse,
            0.48 + 1.0 * amplitude * pulse,
            0.38 + 0.3 * amplitude * pulse,
        )
    )

