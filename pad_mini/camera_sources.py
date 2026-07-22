"""OpenCV tarafindan kullanilabilecek kamera kaynaklarini bulur."""

from pathlib import Path
import platform
from urllib.parse import urlsplit, urlunsplit

import cv2


def discover_camera_sources():
    """(etiket, kaynak) ciftlerinden olusan kamera listesini dondurur."""
    if platform.system() == "Linux":
        sources = _discover_linux_cameras()
        if sources:
            return sources

    return _probe_camera_indices()


def parse_camera_source(selected_value, known_sources):
    """Combobox secimini OpenCV'nin kabul edecegi kaynaga cevirir."""
    if selected_value in known_sources:
        return known_sources[selected_value]

    value = selected_value.strip()
    if value.isdigit():
        return int(value)

    return value


def build_droidcam_url(address):
    """Telefon adresini DroidCam'in MJPEG video adresine donusturur."""
    value = address.strip()
    if not value:
        raise ValueError("Telefon IP adresi boş olamaz.")

    if "://" not in value:
        value = "http://" + value

    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Geçerli bir telefon IP adresi girin.")

    try:
        port = parsed.port or 4747
    except ValueError as error:
        raise ValueError("DroidCam portu geçerli değil.") from error

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = "[" + host + "]"
    network_location = "%s:%d" % (host, port)

    path = parsed.path.rstrip("/")
    if not path or path == "/":
        path = "/video"

    return urlunsplit(
        (parsed.scheme, network_location, path, parsed.query, "")
    )


def _discover_linux_cameras():
    sources = []
    device_paths = sorted(
        Path("/dev").glob("video*"),
        key=_video_device_sort_key,
    )

    for device_path in device_paths:
        device_name = _read_linux_camera_name(device_path.name)
        label = "%s — %s" % (device_path, device_name)
        sources.append((label, str(device_path)))

    return sources


def _video_device_sort_key(path):
    suffix = path.name.removeprefix("video")
    if suffix.isdigit():
        return int(suffix)
    return 9999


def _read_linux_camera_name(device_name):
    name_path = Path("/sys/class/video4linux") / device_name / "name"
    try:
        return name_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "Kamera"


def _probe_camera_indices(maximum_index=5):
    sources = []

    for camera_index in range(maximum_index + 1):
        camera = cv2.VideoCapture(camera_index)
        is_available = camera.isOpened()
        camera.release()

        if is_available:
            sources.append(("Kamera %d" % camera_index, camera_index))

    return sources
