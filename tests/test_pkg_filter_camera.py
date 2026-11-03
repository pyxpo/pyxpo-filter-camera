"""pyxpo-filter-camera: Python API over a fake bridge."""
import asyncio

import pytest

from pyxpo import LocalChannel, component, create_app, use_ref
from pyxpo.module_api import NotSupportedError
from pyxpo.module_testing import FakeBridge
from pyxpo.renderers.mock import MockRenderer

import pyxpo_filter_camera as FC


def run(coro):
    return asyncio.run(coro)


def mount(root, **props):
    ch = LocalChannel()
    r = MockRenderer(ch)
    app = create_app(root, ch, **props)
    return app, r


def test_desktop_and_ios():
    with pytest.raises(NotSupportedError):
        run(FC.request_camera_permissions_async())
    with FakeBridge("ios"):
        with pytest.raises(NotSupportedError):
            run(FC.capabilities_async())


def test_permissions():
    with FakeBridge("android") as fb:
        fb.on("FilterCamera", "getPermission", lambda kind: {"status": "undetermined", "canAskAgain": True})
        fb.on("FilterCamera", "requestPermission", lambda kind: {"status": "granted", "canAskAgain": True})
        p = run(FC.get_camera_permissions_async())
        assert p["status"] == "undetermined" and not p["granted"] and p["canAskAgain"]
        assert run(FC.FilterCamera.requestMicrophonePermissionsAsync())["granted"]
        assert fb.calls[-1] == ("FilterCamera", "requestPermission", ("microphone",))
        assert run(FC.FilterCamera.request_camera_permissions_async())["granted"]
        assert fb.calls[-1] == ("FilterCamera", "requestPermission", ("camera",))


def test_capabilities_and_recover():
    caps = [{"facing": "back", "resolutions": {"1080p": [30, 60]}, "max_zoom": 8, "torch": True}]
    with FakeBridge("android") as fb:
        fb.on("FilterCamera", "capabilities", lambda: caps)
        fb.on("FilterCamera", "recoverRecordings", lambda: None)
        assert run(FC.capabilities_async()) == caps
        assert run(FC.recover_recordings_async()) == []


def test_view_props_and_events():
    got = []

    @component
    def Root():
        return FC.FilterCameraView(facing="front", filter="golden", filter_intensity=0.5, lens="puppy",
                                   beauty={"smooth": 0.6, "slim": 1}, effect="glitch", resolution="4k", fps=60,
                                   zoom=2, on_camera_ready=got.append, on_recording_finished=got.append,
                                   style={"flex": 1})

    app, r = mount(Root)
    (node,) = r.find_all("PxFilterCameraView")
    p = node.props
    assert p["facing"] == "front" and p["filter_intensity"] == 0.5
    assert p["grade"] == FC._grade(FC.GRADES["golden"]) and p["grade"]["contrast"] == FC.GRADES["golden"]["contrast"]
    assert p["beauty"] == {"smooth": 0.6, "brighten": 0.0, "slim": 1.0, "eyes": 0.0}
    assert p["lens"] == "puppy" and p["effect"] == "glitch" and p["resolution"] == "4k" and p["fps"] == 60
    assert p["zoom"] == 2.0 and p["mirror_front"] is True and p["active"] is True
    assert p["on_camera_ready"] is True and p["on_recording_finished"] is True and "on_thermal" not in p
    r.emit("camera_ready", node, {"value": {"width": 3840, "height": 2160}})
    r.emit("recording_finished", node, {"value": {"uri": "file:///v.mp4", "reason": "storage"}})
    assert got == [{"width": 3840, "height": 2160}, {"uri": "file:///v.mp4", "reason": "storage"}]


def test_view_defaults_have_no_zoom():
    @component
    def Root():
        return FC.FilterCameraView()

    app, r = mount(Root)
    p = r.find_all("PxFilterCameraView")[0].props
    assert "zoom" not in p and p["facing"] == "back" and p["grade"] is None and p["stickers"] == [] and p["resolution"] == "1080p"


@pytest.mark.parametrize("kw", [{"facing": "side"}, {"filter": "valencia"}, {"lens": "dog"}, {"effect": "blur"},
                                {"resolution": "8k"}, {"fps": 24}, {"fps": True}, {"zoom": 0.5},
                                {"filter_intensity": 2}, {"beauty": {"smooth": 2}}, {"beauty": {"lips": 1}},
                                {"beauty": "max"}, {"grade": "warm"}, {"grade": {"glow": 1}},
                                {"grade": {"contrast": 3}}, {"grade": {"hue": 200}}, {"grade": {"shadow": (1, 0)}},
                                {"grade": {"shadow": (0, 2, 0)}}, {"grade": {"mono": True}}, {"grade": {"whites": 2}},
                                {"grade": {"sharpen": -0.5}}, {"grade": {"hsl": "red"}},
                                {"grade": {"hsl": {"pink": (0, 0, 0)}}}, {"grade": {"hsl": {"red": (0, 0)}}},
                                {"grade": {"hsl": {"red": (0, 2, 0)}}},
                                {"stickers": "x"}, {"stickers": [{}]}, {"stickers": [{"emoji": ""}]},
                                {"stickers": [{"emoji": "a", "anchor": "chin"}]},
                                {"stickers": [{"emoji": "a", "size": 9}]}, {"stickers": [{"emoji": "a", "spin": 1}]},
                                {"stickers": [{"emoji": "a"}] * 9}])
def test_view_bad_props(kw):
    @component
    def Root():
        return FC.FilterCameraView(**kw)

    with pytest.raises((ValueError, TypeError)):
        mount(Root, on_error=lambda e: (_ for _ in ()).throw(e))


def test_custom_grade_overrides_filter():
    @component
    def Root():
        return FC.FilterCameraView(filter="noir", grade={"saturation": 1.4, "hue": -30, "shadow": (0.1, 0, -0.2)})

    app, r = mount(Root)
    g = r.find_all("PxFilterCameraView")[0].props["grade"]
    assert set(g) == set(FC.GRADE_DEFAULTS)
    assert g["saturation"] == 1.4 and g["hue"] == -30.0 and g["shadow"] == [0.1, 0.0, -0.2]
    assert g["mono"] == 0.0 and g["contrast"] == 1.0 and g["highlight"] == [0.0, 0.0, 0.0]


def test_stickers_fill_defaults():
    @component
    def Root():
        return FC.FilterCameraView(stickers=[{"emoji": "😎", "size": 1}, {"emoji": "⭐", "anchor": "forehead",
                                                                          "dy": 0.3, "rotation": -20}])

    app, r = mount(Root)
    st = r.find_all("PxFilterCameraView")[0].props["stickers"]
    assert st[0] == {"emoji": "😎", "anchor": "eyes", "size": 1.0, "dx": 0.0, "dy": 0.0, "rotation": 0.0}
    assert st[1]["anchor"] == "forehead" and st[1]["dy"] == 0.3 and st[1]["rotation"] == -20.0


def test_sticker_edit_props_and_events():
    got = []

    @component
    def Root():
        return FC.FilterCameraView(stickers=[{"emoji": "😎"}], sticker_edit=True, selected_sticker=0,
                                   on_sticker_press=got.append, on_sticker_change=got.append)

    app, r = mount(Root)
    v = r.find_all("PxFilterCameraView")[0]
    assert v.props["sticker_edit"] is True and v.props["selected_sticker"] == 0
    r.emit("sticker_press", v, {"value": {"index": 0}})
    r.emit("sticker_change", v, {"value": {"index": 0, "dx": 0.1, "dy": -0.2, "size": 1.2, "rotation": 15}})
    assert got == [{"index": 0}, {"index": 0, "dx": 0.1, "dy": -0.2, "size": 1.2, "rotation": 15}]

    @component
    def Plain():
        return FC.FilterCameraView()

    app, r = mount(Plain)
    v = r.find_all("PxFilterCameraView")[0]
    assert v.props["sticker_edit"] is False and v.props["selected_sticker"] == -1


def test_hsl_goes_on_the_wire_as_24_floats_in_colour_order():
    wire = FC._grade({"hsl": {"orange": (0.5, -0.25, 0), "magenta": [0, 0, 1]}})["hsl"]
    assert len(wire) == 24 and wire[3:6] == [0.5, -0.25, 0.0] and wire[21:24] == [0.0, 0.0, 1.0] and sum(wire) == 1.25
    assert FC._grade({})["hsl"] == []
    assert FC._grade({"vibrance": 0.5, "tone_shadows": -1, "sharpen": 1})["tone_shadows"] == -1.0


def test_every_filter_has_a_valid_grade():
    assert set(FC.GRADES) == set(FC.FILTERS) - {"none"}
    for name, g in FC.GRADES.items():
        assert FC._grade(g), name


def test_handle_methods():
    refs = []

    @component
    def Root():
        ref = use_ref(None)
        refs.append(ref)
        return FC.FilterCameraView(ref=ref)

    app, r = mount(Root)  # the app sets the desktop env, so the fake bridge comes after
    with FakeBridge("android") as fb:
        tag = r.find_all("PxFilterCameraView")[0].id
        done = {"uri": "file:///v.mp4", "duration": 61.2, "size": 9000, "segments": 3, "reason": "stopped"}
        fb.on("FilterCamera", "takePhoto", lambda t, o: {"uri": "file:///p.jpg", "width": 1080, "height": 1920})
        fb.on("FilterCamera", "startRecording", lambda t, o: None)
        fb.on("FilterCamera", "pauseRecording", lambda t: None)
        fb.on("FilterCamera", "resumeRecording", lambda t: None)
        fb.on("FilterCamera", "stopRecording", lambda t: done)
        cam = refs[-1].current
        assert isinstance(cam, FC.FilterCameraHandle)
        assert run(cam.take_photo_async(0.8))["uri"] == "file:///p.jpg"
        assert fb.calls[-1] == ("FilterCamera", "takePhoto", (tag, {"quality": 0.8}))
        run(cam.start_recording_async())
        assert fb.calls[-1] == ("FilterCamera", "startRecording",
                                (tag, {"mute": False, "speed": 1.0, "segmentSeconds": 30.0}))
        run(cam.startRecordingAsync(speed=4, max_duration=600, segment_seconds=10))
        assert fb.calls[-1][2][1] == {"mute": True, "speed": 4.0, "segmentSeconds": 10.0, "maxDuration": 600.0}
        run(cam.pause_recording_async())
        assert fb.calls[-1] == ("FilterCamera", "pauseRecording", (tag,))
        run(cam.resume_recording_async())
        assert run(cam.stop_recording_async()) == done
        for bad in ({"speed": 3}, {"segment_seconds": 1}, {"max_duration": 0}):
            with pytest.raises(ValueError):
                run(cam.start_recording_async(**bad))
        with pytest.raises(ValueError):
            run(cam.take_photo_async(quality=2))
