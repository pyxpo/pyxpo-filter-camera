"""A filter camera: live GPU color filters, beauty, face lenses and effects, burned into long video recordings.

    import pyxpo_filter_camera as FilterCamera
    from pyxpo import use_ref

    perm, request, _ = FilterCamera.use_camera_permissions(request=True)
    cam = use_ref(None)

    FilterCamera.FilterCameraView(ref=cam, facing="front", filter="golden", lens="puppy",
                                  beauty={"smooth": 0.6, "brighten": 0.3, "slim": 0.2},
                                  effect="none", resolution="1080p", fps=30, style={"flex": 1})

    await cam.current.start_recording_async(segment_seconds=30)
    await cam.current.pause_recording_async()
    await cam.current.resume_recording_async()
    video = await cam.current.stop_recording_async()
    video["uri"], video["duration"], video["size"]

Everything you see in the preview (filter, beauty, lens, effect) is drawn on the GPU and written into the video
and photos. Recordings are saved in segments while they run, so a crash loses at most one segment:
`recover_recordings_async()` joins what an earlier run left behind.

Android only for now (Camera2, OpenGL ES, MediaCodec, ML Kit face contours with the bundled model).
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from pyxpo import component, use_effect, use_imperative_handle, use_ref, use_state
from pyxpo.module_api import (NotSupportedError, call, native_view, permission_response, platform, require_linked,
                              snake_aliases, view_tag)

__version__ = "0.1.0a1"

__all__ = [
    "FilterCameraView", "FilterCameraHandle", "FilterCamera", "FILTERS", "GRADES", "GRADE_DEFAULTS", "LENSES", "ANCHORS", "STICKER_DEFAULTS", "MAX_STICKERS", "EFFECTS", "SPEEDS",
    "RESOLUTIONS", "PermissionStatus", "get_camera_permissions_async", "request_camera_permissions_async",
    "get_microphone_permissions_async", "request_microphone_permissions_async", "use_camera_permissions",
    "use_microphone_permissions", "capabilities_async", "recover_recordings_async", "NotSupportedError",
]

_View = native_view("PxFilterCameraView")

FILTERS = ("none", "crisp", "breeze", "lunar", "meadow", "faded", "warmth", "hazy", "cream", "bright", "pastel",
           "valley", "cross", "lofi", "honey", "peach", "ash", "ink", "amber", "sunrise", "arctic", "dawn", "ember",
           "seventies", "blush",
           "aurora", "golden", "noir", "silver", "retro", "instant", "film", "dream", "frost", "sunset",
           "vivid", "mint", "rose", "cinema", "dusk", "sepia", "pop")
"""Color filters (`filter=`), all made from one color-grading shader. The first group recalls the classic
social-app photo looks (crisp contrast, faded matte, warm vintage, soft B&W); the rest are Lumo originals."""

GRADE_DEFAULTS: Dict[str, Any] = {
    "exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "temperature": 0.0, "tint": 0.0, "fade": 0.0,
    "shadow": (0.0, 0.0, 0.0), "highlight": (0.0, 0.0, 0.0), "split": 0.0, "vignette": 0.0, "grain": 0.0,
    "mono": 0.0, "sepia": 0.0, "hue": 0.0, "vibrance": 0.0, "tone_highlights": 0.0, "tone_shadows": 0.0, "whites": 0.0,
    "blacks": 0.0, "clarity": 0.0, "dehaze": 0.0, "sharpen": 0.0, "hsl": {},
}
"""The color grade every filter is made of (`grade=`), with its neutral values. `exposure` -1..1, `contrast` and
`saturation` 0..2 (1 = unchanged), `temperature` and `tint` -1..1, `fade`, `vignette`, `grain`, `mono`, `sepia` 0..1,
`hue` a rotation in degrees -180..180, `shadow` and `highlight` (r, g, b) tints -1..1 blended in by `split` 0..1.
`vibrance`, `tone_highlights`, `tone_shadows`, `whites`, `blacks`, `clarity`, `dehaze` -1..1, `sharpen` 0..1. `hsl` tunes
one colour range at a time: {colour: (hue, saturation, luminance)} with each -1..1 and colours from `HSL_COLORS`."""

HSL_COLORS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
"""The colour ranges of `grade["hsl"]`, in hue order."""

_GRADE_RANGES = {"exposure": (-1, 1), "contrast": (0, 2), "saturation": (0, 2), "temperature": (-1, 1),
                 "tint": (-1, 1), "fade": (0, 1), "split": (0, 1), "vignette": (0, 1), "grain": (0, 1),
                 "mono": (0, 1), "sepia": (0, 1), "hue": (-180, 180), "vibrance": (-1, 1), "tone_highlights": (-1, 1),
                 "tone_shadows": (-1, 1), "whites": (-1, 1), "blacks": (-1, 1), "clarity": (-1, 1), "dehaze": (-1, 1),
                 "sharpen": (0, 1)}

GRADES: Dict[str, Dict[str, Any]] = {
    # Classic social-app looks, rebuilt from scratch with this grade (comment = the look they recall).
    "crisp": {"contrast": 1.2, "saturation": 1.35, "shadow": (-0.05, 0.1, 0.3), "highlight": (0.2, 0.15, 0.0),
        "split": 0.12},  # Clarendon
    "breeze": {"exposure": 0.07, "contrast": 0.9, "saturation": 0.9, "hue": -10.0, "fade": 0.2,
        "highlight": (0.15, 0.1, 0.25), "split": 0.1},  # Gingham
    "lunar": {"exposure": 0.14, "contrast": 1.1, "mono": 1.0, "fade": 0.15},  # Moon
    "meadow": {"exposure": 0.08, "contrast": 0.9, "saturation": 0.95, "temperature": -0.15, "tint": 0.1,
        "highlight": (0.05, 0.15, 0.15), "split": 0.1},  # Lark
    "faded": {"exposure": 0.14, "contrast": 0.85, "saturation": 0.75, "sepia": 0.22, "fade": 0.25},  # Reyes
    "warmth": {"exposure": 0.2, "contrast": 1.15, "saturation": 1.6, "sepia": 0.35, "temperature": 0.15},  # Juno
    "hazy": {"exposure": 0.07, "saturation": 0.66, "fade": 0.2, "shadow": (0.15, 0.05, 0.2),
        "highlight": (0.25, 0.2, 0.05), "split": 0.12},  # Slumber
    "cream": {"exposure": 0.2, "contrast": 1.25, "saturation": 0.9, "sepia": 0.5, "hue": -2.0, "fade": 0.1},  # Crema
    "bright": {"exposure": 0.07, "contrast": 1.05, "saturation": 1.5, "vignette": 0.15, "highlight": (0.1, 0.05, 0.0),
        "split": 0.08},  # Ludwig
    "pastel": {"exposure": 0.26, "contrast": 0.9, "saturation": 0.85, "hue": -20.0, "fade": 0.15,
        "highlight": (0.2, 0.1, 0.15), "split": 0.1},  # Aden
    "valley": {"exposure": 0.11, "contrast": 1.08, "sepia": 0.08, "temperature": 0.2, "fade": 0.1,
        "highlight": (0.2, 0.1, 0.0), "split": 0.1},  # Valencia
    "cross": {"contrast": 1.3, "saturation": 1.1, "sepia": 0.3, "vignette": 0.55, "shadow": (0.0, 0.1, 0.25),
        "highlight": (0.25, 0.2, 0.0), "split": 0.15},  # X-Pro II
    "lofi": {"contrast": 1.5, "saturation": 1.1, "vignette": 0.45},  # Lo-Fi
    "honey": {"contrast": 1.5, "saturation": 1.3, "sepia": 0.1, "temperature": 0.3, "vignette": 0.4},  # Hefe
    "peach": {"exposure": 0.07, "contrast": 1.2, "saturation": 1.2, "sepia": 0.2, "tint": -0.15, "fade": 0.12,
        "shadow": (0.0, 0.05, 0.25), "highlight": (0.3, 0.15, 0.1), "split": 0.15},  # Nashville
    "ash": {"exposure": 0.13, "contrast": 0.95, "mono": 1.0, "fade": 0.2, "sepia": 0.1},  # Willow
    "ink": {"exposure": 0.14, "contrast": 1.1, "mono": 1.0},  # Inkwell
    "amber": {"exposure": 0.14, "contrast": 0.9, "saturation": 1.5, "hue": -10.0, "fade": 0.12,
        "highlight": (0.15, 0.1, 0.2), "split": 0.1},  # Amaro
    "sunrise": {"exposure": 0.07, "contrast": 0.9, "saturation": 0.9, "sepia": 0.2, "temperature": 0.2, "fade": 0.15,
        "highlight": (0.3, 0.2, 0.1), "split": 0.1},  # Rise
    "arctic": {"exposure": 0.2, "contrast": 0.9, "saturation": 1.1, "temperature": -0.4, "vignette": 0.2,
        "highlight": (0.0, 0.1, 0.25), "split": 0.12},  # Hudson
    "dawn": {"contrast": 0.9, "sepia": 0.2, "temperature": 0.25, "fade": 0.2, "vignette": 0.35,
        "highlight": (0.3, 0.2, 0.1), "split": 0.1},  # Earlybird
    "ember": {"exposure": -0.15, "contrast": 1.5, "temperature": 0.4, "tint": -0.1, "vignette": 0.5,
        "shadow": (0.2, 0.0, 0.15), "split": 0.15},  # Toaster
    "seventies": {"exposure": 0.14, "contrast": 1.1, "saturation": 1.3, "tint": -0.25, "fade": 0.15,
        "highlight": (0.3, 0.05, 0.1), "split": 0.15},  # 1977
    "blush": {"contrast": 1.1, "saturation": 1.1, "tint": -0.15, "vignette": 0.25, "highlight": (0.25, 0.15, 0.15),
        "split": 0.12},  # Mayfair
    # Lumo originals.
    "aurora": {"contrast": 1.06, "saturation": 1.15, "temperature": -0.2, "shadow": (-0.1, 0.25, 0.35),
        "highlight": (0.35, 0.05, 0.3), "split": 0.12, "vignette": 0.2},
    "golden": {"exposure": 0.05, "contrast": 1.05, "saturation": 1.1, "temperature": 0.55, "tint": 0.05,
        "shadow": (0.15, 0.05, -0.1), "highlight": (0.3, 0.2, 0.0), "split": 0.1, "vignette": 0.25},
    "noir": {"contrast": 1.35, "mono": 1.0, "vignette": 0.45, "grain": 0.25, "exposure": -0.05},
    "silver": {"contrast": 0.95, "mono": 1.0, "fade": 0.25, "exposure": 0.08, "grain": 0.1},
    "retro": {"contrast": 0.92, "saturation": 0.8, "temperature": 0.3, "fade": 0.35, "shadow": (0.1, 0.15, 0.05),
        "highlight": (0.25, 0.15, -0.05), "split": 0.12, "vignette": 0.3, "grain": 0.2},
    "instant": {"exposure": 0.1, "contrast": 0.9, "saturation": 0.85, "temperature": 0.15, "tint": -0.1, "fade": 0.3,
        "highlight": (0.1, 0.2, 0.15), "split": 0.1, "vignette": 0.15},
    "film": {"contrast": 1.1, "saturation": 0.9, "shadow": (0.0, 0.15, 0.2), "highlight": (0.25, 0.12, 0.0),
        "split": 0.15, "fade": 0.12, "grain": 0.3, "vignette": 0.2},
    "dream": {"exposure": 0.15, "contrast": 0.85, "saturation": 0.9, "tint": 0.1, "fade": 0.2,
        "shadow": (0.2, 0.1, 0.3), "highlight": (0.3, 0.2, 0.3), "split": 0.1},
    "frost": {"contrast": 1.05, "saturation": 0.8, "temperature": -0.6, "exposure": 0.05,
        "highlight": (0.0, 0.1, 0.25), "split": 0.1},
    "sunset": {"contrast": 1.08, "saturation": 1.2, "temperature": 0.45, "tint": 0.15, "shadow": (0.25, -0.05, 0.2),
        "highlight": (0.35, 0.15, -0.1), "split": 0.15, "vignette": 0.25},
    "vivid": {"contrast": 1.15, "saturation": 1.45, "exposure": 0.03},
    "mint": {"contrast": 0.98, "saturation": 0.95, "temperature": -0.1, "tint": 0.3, "fade": 0.12,
        "highlight": (0.05, 0.2, 0.1), "split": 0.1},
    "rose": {"exposure": 0.05, "contrast": 0.97, "saturation": 1.05, "temperature": 0.1, "tint": -0.35,
        "highlight": (0.3, 0.1, 0.15), "split": 0.12, "fade": 0.08},
    "cinema": {"contrast": 1.2, "saturation": 0.9, "shadow": (-0.05, 0.15, 0.25), "highlight": (0.3, 0.15, -0.05),
        "split": 0.18, "vignette": 0.35},
    "dusk": {"exposure": -0.1, "contrast": 1.05, "saturation": 0.9, "temperature": -0.2, "tint": -0.15,
        "shadow": (0.1, 0.0, 0.3), "highlight": (0.3, 0.1, 0.1), "split": 0.15, "vignette": 0.35},
    "sepia": {"contrast": 1.0, "mono": 1.0, "shadow": (0.15, 0.05, -0.1), "highlight": (0.3, 0.18, 0.0),
        "split": 0.22, "fade": 0.12, "vignette": 0.3},
    "pop": {"contrast": 1.25, "saturation": 1.6, "exposure": 0.05, "vignette": 0.1},
}
"""The grade behind every `FILTERS` name except "none". Copy one as a start for your own `grade=`."""

LENSES = ("none", "puppy", "kitty", "bunny", "shades", "hearts", "crown", "devil", "halo", "masquerade",
          "mustache", "clown")
"""Face lenses (`lens=`): stickers that follow the face."""

ANCHORS = ("eyes", "nose", "mouth", "forehead")
"""Where a sticker of your own (`stickers=`) sits on the face."""

STICKER_DEFAULTS: Dict[str, Any] = {"anchor": "eyes", "size": 0.4, "dx": 0.0, "dy": 0.0, "rotation": 0.0}
"""The keys of one `stickers=` entry besides `emoji`. `size`, `dx` (toward the screen-right eye) and `dy` (toward
the forehead) are in face widths; `rotation` is degrees, clockwise."""

_STICKER_RANGES = {"size": (0.05, 3.0), "dx": (-2.0, 2.0), "dy": (-2.0, 2.0), "rotation": (-180.0, 180.0)}
MAX_STICKERS = 8

EFFECTS = ("none", "glitch", "rgb_split", "mirror", "vhs", "pixelate", "kaleido", "thermal", "comic", "sketch",
           "neon", "posterize", "bulge", "wave")
"""Effects (`effect=`). Some move over time (glitch, vhs, wave)."""

SPEEDS = (0.5, 1, 2, 4, 8)
"""Recording speeds: 0.5 is slow motion (best with `fps=60`), 2-8 is timelapse. Only speed 1 records sound."""

RESOLUTIONS = ("720p", "1080p", "4k")

_BEAUTY = ("smooth", "brighten", "slim", "eyes")
_FACING = ("front", "back")


class PermissionStatus:
    GRANTED = "granted"
    DENIED = "denied"
    UNDETERMINED = "undetermined"


def _native() -> None:
    if platform() == "desktop":
        raise NotSupportedError("the filter camera needs Android")
    if platform() == "ios":
        raise NotSupportedError("pyxpo-filter-camera supports Android only for now")
    require_linked("FilterCamera", "filter-camera", package="pyxpo-filter-camera")


def _choice(name: str, value: Any, allowed: Tuple[Any, ...]) -> Any:
    if isinstance(value, bool) or value not in allowed:
        raise ValueError("%s must be one of %s, got %r" % (name, ", ".join(str(a) for a in allowed), value))
    return value


def _unit(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError("%s must be a number from 0 to 1, got %r" % (name, value))
    return float(value)


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("%s must be a number > 0, got %r" % (name, value))
    return float(value)


def _opt(o: Dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in o:
            return o[n]
    return None


def _beauty(b: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if b is None:
        b = {}
    if not isinstance(b, dict):
        raise TypeError("beauty must be a dict, got %s" % type(b).__name__)
    for k in b:
        if k not in _BEAUTY:
            raise ValueError("beauty keys are %s, got %r" % (", ".join(_BEAUTY), k))
    return {k: _unit("beauty[%r]" % k, b.get(k, 0)) for k in _BEAUTY}


def _grade(g: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A full grade dict for the wire (missing keys neutral), or None for no color pass."""
    if g is None:
        return None
    if not isinstance(g, dict):
        raise TypeError("grade must be a dict, got %s" % type(g).__name__)
    out = dict(GRADE_DEFAULTS)
    for k, v in g.items():
        if k not in GRADE_DEFAULTS:
            raise ValueError("grade keys are %s, got %r" % (", ".join(GRADE_DEFAULTS), k))
        if k in ("shadow", "highlight"):
            if not isinstance(v, (list, tuple)) or len(v) != 3 or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not -1 <= x <= 1 for x in v):
                raise ValueError("grade[%r] must be (r, g, b), each -1 to 1, got %r" % (k, v))
            out[k] = [float(x) for x in v]
            continue
        if k == "hsl":
            out[k] = _hsl(v)
            continue
        lo, hi = _GRADE_RANGES[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise ValueError("grade[%r] must be a number from %s to %s, got %r" % (k, lo, hi, v))
        out[k] = float(v)
    out["shadow"], out["highlight"] = list(out["shadow"]), list(out["highlight"])
    if "hsl" not in g:
        out["hsl"] = []
    return out


def _hsl(v: Any) -> List[float]:
    """`grade["hsl"]` for the wire: 24 floats, (hue, saturation, luminance) per colour in `HSL_COLORS` order."""
    if not isinstance(v, dict):
        raise ValueError("grade['hsl'] must be a dict of colour to (hue, saturation, luminance), got %r" % (v,))
    for name, triple in v.items():
        if name not in HSL_COLORS:
            raise ValueError("grade['hsl'] colours are %s, got %r" % (", ".join(HSL_COLORS), name))
        if not isinstance(triple, (list, tuple)) or len(triple) != 3 or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not -1 <= x <= 1 for x in triple):
            raise ValueError("grade['hsl'][%r] must be (hue, saturation, luminance), each -1 to 1, got %r"
                             % (name, triple))
    return [float(x) for name in HSL_COLORS for x in v.get(name, (0.0, 0.0, 0.0))]


def _stickers(items: Any) -> List[Dict[str, Any]]:
    """Full sticker dicts for the wire (missing keys take `STICKER_DEFAULTS`)."""
    if items is None:
        return []
    if not isinstance(items, (list, tuple)):
        raise TypeError("stickers must be a list of dicts, got %s" % type(items).__name__)
    if len(items) > MAX_STICKERS:
        raise ValueError("at most %d stickers, got %d" % (MAX_STICKERS, len(items)))
    out = []
    for i, st in enumerate(items):
        if not isinstance(st, dict):
            raise TypeError("stickers[%d] must be a dict, got %s" % (i, type(st).__name__))
        emoji = st.get("emoji")
        if not isinstance(emoji, str) or not 0 < len(emoji) <= 16:
            raise ValueError("stickers[%d]['emoji'] must be a short non-empty string, got %r" % (i, emoji))
        one: Dict[str, Any] = dict(STICKER_DEFAULTS, emoji=emoji)
        for k, v in st.items():
            if k == "emoji":
                continue
            if k == "anchor":
                one[k] = _choice("stickers[%d]['anchor']" % i, v, ANCHORS)
                continue
            if k not in _STICKER_RANGES:
                raise ValueError("sticker keys are emoji, %s, got %r" % (", ".join(STICKER_DEFAULTS), k))
            lo, hi = _STICKER_RANGES[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ValueError("stickers[%d][%r] must be a number from %s to %s, got %r" % (i, k, lo, hi, v))
            one[k] = float(v)
        out.append(one)
    return out


# permissions

def _permission(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = raw or {}
    return permission_response(str(raw.get("status", "denied")), bool(raw.get("canAskAgain", False)))


async def get_camera_permissions_async() -> Dict[str, Any]:
    """The camera permission, without asking (Expo's `PermissionResponse`)."""
    _native()
    return _permission(await call("FilterCamera", "getPermission", "camera", reply=True))


async def request_camera_permissions_async() -> Dict[str, Any]:
    """Asks for the camera."""
    _native()
    return _permission(await call("FilterCamera", "requestPermission", "camera", reply=True))


async def get_microphone_permissions_async() -> Dict[str, Any]:
    """The microphone permission (videos with sound), without asking."""
    _native()
    return _permission(await call("FilterCamera", "getPermission", "microphone", reply=True))


async def request_microphone_permissions_async() -> Dict[str, Any]:
    """Asks for the microphone."""
    _native()
    return _permission(await call("FilterCamera", "requestPermission", "microphone", reply=True))


def _permission_hook(getter: Callable, requester: Callable, get: bool, request: bool) -> Tuple[Any, Callable, Callable]:
    status, set_status = use_state(None)

    async def ask() -> Dict[str, Any]:
        res = await requester()
        set_status(res)
        return res

    async def read() -> Dict[str, Any]:
        res = await getter()
        set_status(res)
        return res

    def start() -> None:
        if request:
            asyncio.ensure_future(ask())
        elif get:
            asyncio.ensure_future(read())

    use_effect(start, [])
    return status, ask, read


def use_camera_permissions(get: bool = True, request: bool = False) -> Tuple[Any, Callable, Callable]:
    """`(response, request, get)`: the camera permission (None until read), and coroutines that ask for it or read
    it again and update the component. `get=True` reads it on mount; `request=True` asks on mount."""
    return _permission_hook(get_camera_permissions_async, request_camera_permissions_async, get, request)


def use_microphone_permissions(get: bool = True, request: bool = False) -> Tuple[Any, Callable, Callable]:
    """Like `use_camera_permissions`, for the microphone."""
    return _permission_hook(get_microphone_permissions_async, request_microphone_permissions_async, get, request)


async def capabilities_async() -> List[Dict[str, Any]]:
    """What each camera can record: a list of `{"facing", "resolutions": {"720p": [30, 60], ...}, "max_zoom",
    "torch"}`. A resolution is listed with the frame rates both the camera and the video encoder support."""
    _native()
    return list(await call("FilterCamera", "capabilities", reply=True) or [])


async def recover_recordings_async() -> List[Dict[str, Any]]:
    """Joins recordings that an earlier run did not finish (the app was killed or crashed while recording) and
    returns them like `stop_recording_async()` results, plus `"recovered": True`. A segment that was being written
    at the crash is lost; the others are kept. Call it once at start, before recording."""
    _native()
    return list(await call("FilterCamera", "recoverRecordings", reply=True) or [])


@snake_aliases
class FilterCamera:
    """camelCase names of the module functions."""
    getCameraPermissionsAsync = staticmethod(get_camera_permissions_async)
    requestCameraPermissionsAsync = staticmethod(request_camera_permissions_async)
    getMicrophonePermissionsAsync = staticmethod(get_microphone_permissions_async)
    requestMicrophonePermissionsAsync = staticmethod(request_microphone_permissions_async)
    capabilitiesAsync = staticmethod(capabilities_async)
    recoverRecordingsAsync = staticmethod(recover_recordings_async)


# the view

class FilterCameraHandle:
    """`ref.current` of a mounted `FilterCameraView`."""

    def __init__(self, view_ref: Any) -> None:
        self._view = view_ref

    def _tag(self) -> int:
        _native()
        tag = view_tag(self._view)
        if tag is None:
            raise RuntimeError("the FilterCameraView is not mounted")
        return tag

    async def take_photo_async(self, quality: float = 0.95) -> Dict[str, Any]:
        """A JPEG of what the preview shows, at the full frame size: `{"uri", "width", "height"}` (app cache)."""
        tag = self._tag()
        return await call("FilterCamera", "takePhoto", tag, {"quality": _unit("quality", quality)}, reply=True)

    async def start_recording_async(self, mute: bool = False, speed: float = 1, max_duration: Optional[float] = None,
                                    segment_seconds: float = 30) -> None:
        """Starts recording what the preview shows. Returns once recording runs.

        `speed` is one of `SPEEDS`; only speed 1 records sound, and sound needs the microphone permission unless
        `mute=True`. `max_duration` stops after that many seconds of video. The video is written in segments of
        `segment_seconds` and joined at the end. Recording stops by itself when the app goes to the background or
        storage runs low; `on_recording_finished` then gets the result with a `"reason"`."""
        tag = self._tag()
        args: Dict[str, Any] = {
            "mute": bool(mute) or speed != 1,
            "speed": float(_choice("speed", speed, SPEEDS)),
            "segmentSeconds": _positive("segment_seconds", segment_seconds),
        }
        if args["segmentSeconds"] < 2:
            raise ValueError("segment_seconds must be at least 2, got %r" % segment_seconds)
        if max_duration is not None:
            args["maxDuration"] = _positive("max_duration", max_duration)
        await call("FilterCamera", "startRecording", tag, args, reply=True)

    async def pause_recording_async(self) -> None:
        """Pauses; the paused time is left out of the video."""
        await call("FilterCamera", "pauseRecording", self._tag(), reply=True)

    async def resume_recording_async(self) -> None:
        await call("FilterCamera", "resumeRecording", self._tag(), reply=True)

    async def stop_recording_async(self) -> Optional[Dict[str, Any]]:
        """Stops and joins the segments: `{"uri", "duration", "size", "width", "height", "segments", "reason"}`, an
        `.mp4` in the app cache. None when nothing was recording."""
        return await call("FilterCamera", "stopRecording", self._tag(), reply=True)

    # camelCase names
    takePhotoAsync = take_photo_async
    startRecordingAsync = start_recording_async
    pauseRecordingAsync = pause_recording_async
    resumeRecordingAsync = resume_recording_async
    stopRecordingAsync = stop_recording_async


_EVENTS = ("on_camera_ready", "on_mount_error", "on_recording_progress", "on_recording_finished", "on_thermal",
           "on_zoom_change", "on_face", "on_double_tap", "on_sticker_press", "on_sticker_change")


@component
def FilterCameraView(facing: str = "back", filter: str = "none", filter_intensity: float = 1,
                     grade: Optional[Dict[str, Any]] = None, beauty: Optional[Dict[str, Any]] = None, lens: str = "none",
                     stickers: Optional[List[Dict[str, Any]]] = None, sticker_edit: bool = False,
                     selected_sticker: int = -1, effect: str = "none",
                     effect_intensity: float = 1, resolution: str = "1080p", fps: int = 30,
                     stabilization: bool = True, torch: bool = False, zoom: Optional[float] = None,
                     mirror_front: bool = True, active: bool = True, style: Any = None, **props: Any):
    """A live filtered camera preview sized by `style` (it fills the view and crops the edges). `ref.current` is
    a `FilterCameraHandle`.

    `filter` is one of `FILTERS`. `grade` is your own filter: a dict of `GRADE_DEFAULTS` keys (missing ones stay
    neutral); when set it is used instead of `filter`. `lens` one of `LENSES`, `effect` one of `EFFECTS`.
    `stickers` is your own lens, drawn over `lens`: up to `MAX_STICKERS` dicts with an `emoji` (any text) plus
    `STICKER_DEFAULTS` keys, e.g. `[{"emoji": "😎", "anchor": "eyes", "size": 1}]`. `beauty` has `smooth`,
    `brighten`, `slim` and `eyes`, each 0-1 (slim and eyes need a face). `resolution` (`RESOLUTIONS`) and `fps`
    (30 or 60) are the recording size; when the camera cannot do it, the view uses the closest it can and says so
    in `on_camera_ready`. Changes to facing, resolution or fps wait until a recording ends.

    `zoom` is a zoom ratio (1 = none); leave it None to let the user pinch, and get the ratio in `on_zoom_change`.
    Tapping focuses there. `mirror_front` shows and records the front camera like a mirror.

    `sticker_edit` turns touches into sticker editing (no zoom, focus or double tap): a tap on one of `stickers`
    calls `on_sticker_press({"index"})` (-1 for a tap on none); one finger drags the sticker under it, or
    `selected_sticker` when it is on none; two fingers also pinch to size and twist to turn. The sticker moves live;
    when the fingers lift, `on_sticker_change({"index", "dx", "dy", "size", "rotation"})` gives its new place, and
    the app should put it in `stickers`.

    Events: `on_camera_ready({"width", "height", "fps", "facing", "resolution", "min_zoom", "max_zoom"})`,
    `on_mount_error({"message"})`, `on_recording_progress({"duration", "size", "fps", "segments", "paused"})` every
    second, `on_recording_finished(result)`, `on_thermal({"status"})` (`"none"` to `"shutdown"`),
    `on_zoom_change({"zoom"})`, `on_face({"found"})` when a face appears or goes, `on_double_tap({"x", "y"})`,
    `on_sticker_press` and `on_sticker_change` (see `sticker_edit`)."""
    ref = props.pop("ref", None)
    inner = use_ref(None)
    use_imperative_handle(ref, lambda: FilterCameraHandle(inner), [])
    wire: Dict[str, Any] = {
        "facing": _choice("facing", facing, _FACING),
        "grade": _grade(grade if grade is not None else GRADES.get(_choice("filter", filter, FILTERS))),
        "filter_intensity": _unit("filter_intensity", filter_intensity),
        "beauty": _beauty(beauty),
        "lens": _choice("lens", lens, LENSES),
        "stickers": _stickers(stickers),
        "sticker_edit": bool(sticker_edit),
        "selected_sticker": int(selected_sticker),
        "effect": _choice("effect", effect, EFFECTS),
        "effect_intensity": _unit("effect_intensity", effect_intensity),
        "resolution": _choice("resolution", resolution, RESOLUTIONS),
        "fps": _choice("fps", fps, (30, 60)),
        "stabilization": bool(stabilization),
        "torch": bool(torch),
        "mirror_front": bool(mirror_front),
        "active": bool(active),
    }
    if zoom is not None:
        if isinstance(zoom, bool) or not isinstance(zoom, (int, float)) or zoom < 1:
            raise ValueError("zoom must be a ratio >= 1, got %r" % (zoom,))
        wire["zoom"] = float(zoom)
    for name in _EVENTS:
        fn = props.pop(name, None)
        if fn is not None:
            wire[name] = fn
    return _View(ref=inner, style=style, **wire, **props)
