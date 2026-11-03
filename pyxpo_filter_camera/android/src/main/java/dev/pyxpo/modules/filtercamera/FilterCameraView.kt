package dev.pyxpo.modules.filtercamera

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.MeteringRectangle
import android.media.ImageReader
import android.net.Uri
import android.opengl.EGLSurface
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.util.Range
import android.util.Size
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager
import dev.pyxpo.moduleapi.PxNativeView
import dev.pyxpo.moduleapi.PxReply
import org.json.JSONObject
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * One `FilterCameraView`: Camera2 frames go into a GL texture, through the filter chain, and out to the screen and,
 * while recording, to the video encoder. The camera runs while the view is active and its activity is in front.
 *
 * Threads: the GL thread owns EGL, the camera device and session, and draws each frame; the analysis thread feeds
 * small frames to the face tracker; the worker saves photos and joins recordings. Props and events live on the
 * main thread.
 */
@SuppressLint("ViewConstructor")
class FilterCameraView(private val activity: Activity, private val emit: (String, Any?) -> Unit) : PxNativeView {
    private val surfaceView = SurfaceView(activity)
    override val view = surfaceView

    private val main = Handler(Looper.getMainLooper())
    private val work: ExecutorService = Executors.newSingleThreadExecutor()
    private val glThread = HandlerThread("FilterCamera.gl").apply { start() }
    private val gl = Handler(glThread.looper)
    private val analysisThread = HandlerThread("FilterCamera.analysis").apply { start() }
    private val analysis = Handler(analysisThread.looper)
    private val manager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    // props (main thread)
    private var facing = "back"
    private var resolution = "1080p"
    private var fps = 30
    private var active = true
    private var zoomProp: Float? = null
    private var foreground = true
    private var disposed = false

    // read by the GL thread
    @Volatile private var look = Look()
    @Volatile private var mirrorFront = true
    @Volatile private var stabilization = true
    @Volatile private var torch = false
    @Volatile private var zoom = 1f
    @Volatile private var viewW = 0
    @Volatile private var viewH = 0

    // recording (main thread)
    private var recorder: VideoRecorder? = null
    private var stopping = false
    private val stopReplies = ArrayList<PxReply>()

    private var thermal: Any? = null
    private var lastFound: Boolean? = null

    // sticker editing (main thread): the sticker_edit and selected_sticker props, and the gesture in progress
    private var stickerEdit = false
    private var selected = -1
    private var edit: Edit? = null

    /** Where the app's stickers were last drawn, for touches: [GlRenderer.hits] with the face and frame size. */
    private class Geo(val face: Face, val frameW: Int, val frameH: Int, val hits: List<FloatArray>)
    @Volatile private var geo: Geo? = null

    /**
     * One edit gesture on sticker [index]: [base] is its spec when the current finger layout began, at fingers
     * [x0], [y0] (the midpoint for two), [dist0] apart at [angle0] radians.
     */
    private class Edit(val index: Int, val downX: Float, val downY: Float, val downMs: Long) {
        var base: StickerSpec? = null
        var x0 = 0f
        var y0 = 0f
        var dist0 = 0f
        var angle0 = 0f
        var moved = false
        var changed = false
    }

    private class Config(val facing: String, val resolution: String, val fps: Int) {
        override fun equals(other: Any?) = other is Config && other.facing == facing &&
            other.resolution == resolution && other.fps == fps
        override fun hashCode() = facing.hashCode() * 31 + resolution.hashCode() * 7 + fps
    }

    private class Photo(val quality: Float, val reply: PxReply)

    // GL thread state
    private var egl: Egl? = null
    private var renderer: GlRenderer? = null
    private var texture: SurfaceTexture? = null
    private var textureSurface: Surface? = null
    private var display: EGLSurface? = null
    private var encoder: EGLSurface? = null
    private var encoding: VideoRecorder? = null
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var tracker: FaceTracker? = null
    private var current: Config? = null
    private var info: Info? = null
    private var generation = 0
    private var frameW = 0
    private var frameH = 0
    private var startNs = -1L
    private var focus: MeteringRectangle? = null
    private var focusToken = 0
    private var photo: Photo? = null
    private val stMatrix = FloatArray(16)
    @Volatile private var faceRaw: FloatArray? = null
    @Volatile private var faceSeenNs = 0L
    private var faceSmooth: FloatArray? = null
    private var faceFound = false

    /** What the open camera can do. */
    private class Info(
        val id: String, val front: Boolean, val sensorOrientation: Int, val realtime: Boolean,
        val preview: Size, val fps: Int, val fpsRange: Range<Int>?, val resolution: String,
        val active: Rect, val minZoom: Float, val maxZoom: Float, val torch: Boolean,
        val videoStab: Boolean, val previewStab: Boolean, val ois: Boolean,
    )

    private val callbacks = object : Application.ActivityLifecycleCallbacks {
        override fun onActivityResumed(a: Activity) { if (a === activity) { foreground = true; sync() } }
        override fun onActivityPaused(a: Activity) { if (a === activity) { foreground = false; sync() } }
        override fun onActivityCreated(a: Activity, b: Bundle?) {}
        override fun onActivityStarted(a: Activity) {}
        override fun onActivityStopped(a: Activity) {}
        override fun onActivitySaveInstanceState(a: Activity, b: Bundle) {}
        override fun onActivityDestroyed(a: Activity) {}
    }

    init {
        activity.application.registerActivityLifecycleCallbacks(callbacks)
        gl.post {
            try {
                val e = Egl()
                egl = e
                val r = GlRenderer()
                renderer = r
                val st = SurfaceTexture(r.oesTex)
                st.setOnFrameAvailableListener({ frame() }, gl)
                texture = st
                textureSurface = Surface(st)
                tracker = FaceTracker(::onFace)
            } catch (e: Exception) {
                post { emit("mount_error", mapOf("message" to (e.message ?: "The camera could not start"))) }
            }
        }
        surfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                val s = holder.surface
                gl.post { attachDisplay(s) }
            }
            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                viewW = width
                viewH = height
            }
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                val done = CountDownLatch(1)
                gl.post { detachDisplay(); done.countDown() }
                done.await(1, TimeUnit.SECONDS)
            }
        })
        setUpGestures()
        if (Build.VERSION.SDK_INT >= 29) {
            val pm = activity.getSystemService(Context.POWER_SERVICE) as PowerManager
            val l = PowerManager.OnThermalStatusChangedListener { s -> emit("thermal", mapOf("status" to thermalName(s))) }
            pm.addThermalStatusListener(activity.mainExecutor, l)
            thermal = l
        }
        sync()
    }

    // props

    override fun update(props: JSONObject, removed: Set<String>) {
        if (props.has("facing")) facing = props.getString("facing")
        if (props.has("resolution")) resolution = props.getString("resolution")
        if (props.has("fps")) fps = props.getInt("fps")
        if (props.has("active")) active = props.getBoolean("active")
        if (props.has("mirror_front")) mirrorFront = props.getBoolean("mirror_front")
        var request = false
        if (props.has("stabilization")) { stabilization = props.getBoolean("stabilization"); request = true }
        if (props.has("torch")) { torch = props.getBoolean("torch"); request = true }
        if (props.has("zoom")) {
            val z = props.getDouble("zoom").toFloat()
            zoomProp = z
            zoom = clampZoom(z)
            request = true
        } else if ("zoom" in removed) {
            zoomProp = null
        }
        var l = look
        if (props.has("grade") || "grade" in removed) l = l.copy(grade = props.optJSONObject("grade")?.let(Grade::from))
        if (props.has("filter_intensity")) l = l.copy(filterIntensity = props.getDouble("filter_intensity").toFloat())
        if (props.has("beauty")) {
            val b = props.optJSONObject("beauty") ?: JSONObject()
            l = l.copy(
                smooth = b.optDouble("smooth", 0.0).toFloat(), brighten = b.optDouble("brighten", 0.0).toFloat(),
                slim = b.optDouble("slim", 0.0).toFloat(), eyes = b.optDouble("eyes", 0.0).toFloat(),
            )
        }
        if (props.has("lens")) l = l.copy(lens = props.getString("lens"))
        if (props.has("sticker_edit")) stickerEdit = props.getBoolean("sticker_edit")
        if (props.has("selected_sticker")) selected = props.getInt("selected_sticker")
        if (!stickerEdit) edit = null
        // while a finger moves a sticker, the drawn spec is the live one; the app gets it when the finger lifts
        if ((props.has("stickers") || "stickers" in removed) && edit?.changed != true) {
            val arr = props.optJSONArray("stickers")
            l = l.copy(stickers = if (arr == null) emptyList() else List(arr.length()) { StickerSpec.from(arr.getJSONObject(it)) })
        }
        if (props.has("effect")) l = l.copy(effect = props.getString("effect"))
        if (props.has("effect_intensity")) l = l.copy(effectIntensity = props.getDouble("effect_intensity").toFloat())
        look = l
        if (request) gl.post { repeat() }
        sync()
    }

    /** Opens, reopens or closes the camera to match the props, the activity and the recording. */
    private fun sync() {
        if (disposed) return
        val run = active && foreground
        if (!run && recorder != null) {
            stop(if (!foreground) "background" else "inactive", null)
            return
        }
        // facing, size and fps changes wait for the recording to end
        if (recorder != null || stopping) return
        val want = if (run) Config(facing, resolution, fps) else null
        gl.post { apply(want) }
    }

    // gestures

    @SuppressLint("ClickableViewAccessibility")
    private fun setUpGestures() {
        val taps = GestureDetector(activity, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent) = true
            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                val x = e.x
                val y = e.y
                gl.post { focusAt(x, y) }
                return true
            }
            override fun onDoubleTap(e: MotionEvent): Boolean {
                val d = activity.resources.displayMetrics.density
                emit("double_tap", mapOf("x" to e.x / d, "y" to e.y / d))
                return true
            }
        })
        val pinch = ScaleGestureDetector(activity, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d: ScaleGestureDetector): Boolean {
                if (zoomProp != null) return false
                val z = clampZoom(zoom * d.scaleFactor)
                if (abs(z - zoom) > 0.001f) {
                    zoom = z
                    gl.post { repeat() }
                    emit("zoom_change", mapOf("zoom" to z.toDouble()))
                }
                return true
            }
        })
        surfaceView.setOnTouchListener { _, e ->
            if (stickerEdit) {
                editTouch(e)
            } else {
                pinch.onTouchEvent(e)
                if (!pinch.isInProgress) taps.onTouchEvent(e)
            }
            true
        }
    }

    /**
     * Sticker editing: a tap picks the sticker under it (`sticker_press`); one finger drags the sticker under it, or
     * the selected one; two fingers also size it (pinch) and turn it (twist). Lifting sends `sticker_change`.
     */
    private fun editTouch(e: MotionEvent) {
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                val hit = hitTest(e.x, e.y)
                val ed = Edit(if (hit >= 0) hit else selected, e.x, e.y, SystemClock.uptimeMillis())
                edit = ed
                rebase(ed, e)
            }
            MotionEvent.ACTION_POINTER_DOWN, MotionEvent.ACTION_POINTER_UP -> edit?.let { rebase(it, e) }
            MotionEvent.ACTION_MOVE -> {
                val ed = edit ?: return
                val base = ed.base ?: return
                val g = geo ?: return
                val (mx, my) = mid(e, -1)
                val slop = 8 * activity.resources.displayMetrics.density
                if (!ed.moved && e.pointerCount < 2 && hypot(mx - ed.downX, my - ed.downY) < slop) return
                ed.moved = true
                var spec = moveBy(base, g, mx - ed.x0, my - ed.y0)
                if (e.pointerCount >= 2 && ed.dist0 > 0f) {
                    val dist = hypot(e.getX(1) - e.getX(0), e.getY(1) - e.getY(0))
                    val turn = Math.toDegrees((angle(e) - ed.angle0).toDouble()).toFloat()
                    spec = spec.copy(size = (base.size * dist / ed.dist0).coerceIn(0.05f, 3f), rotation = wrap(base.rotation + turn))
                }
                put(ed.index, spec)
                ed.changed = true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                val ed = edit ?: return
                edit = null
                if (ed.changed) {
                    val s = look.stickers.getOrNull(ed.index) ?: return
                    emit("sticker_change", mapOf(
                        "index" to ed.index, "dx" to s.dx.toDouble(), "dy" to s.dy.toDouble(),
                        "size" to s.size.toDouble(), "rotation" to s.rotation.toDouble(),
                    ))
                } else if (!ed.moved && e.actionMasked == MotionEvent.ACTION_UP &&
                    SystemClock.uptimeMillis() - ed.downMs < 400) {
                    emit("sticker_press", mapOf("index" to hitTest(e.x, e.y)))
                }
            }
        }
    }

    /** Starts the fingers now down (leaving out the one lifting) from the sticker as it is now. */
    private fun rebase(ed: Edit, e: MotionEvent) {
        ed.base = look.stickers.getOrNull(ed.index)
        val lifting = if (e.actionMasked == MotionEvent.ACTION_POINTER_UP) e.actionIndex else -1
        val (mx, my) = mid(e, lifting)
        ed.x0 = mx
        ed.y0 = my
        val down = (0 until e.pointerCount).filter { it != lifting }
        if (down.size >= 2) {
            ed.dist0 = hypot(e.getX(down[1]) - e.getX(down[0]), e.getY(down[1]) - e.getY(down[0]))
            ed.angle0 = atan2(e.getY(down[1]) - e.getY(down[0]), e.getX(down[1]) - e.getX(down[0]))
        } else {
            ed.dist0 = 0f
        }
    }

    /** The middle of the first two fingers down, leaving out pointer [skip]. */
    private fun mid(e: MotionEvent, skip: Int): Pair<Float, Float> {
        val down = (0 until e.pointerCount).filter { it != skip }.take(2)
        if (down.isEmpty()) return Pair(e.x, e.y)
        return Pair(down.map { e.getX(it) }.average().toFloat(), down.map { e.getY(it) }.average().toFloat())
    }

    /** The angle of the line from finger 0 to finger 1, on screen (y down, so it grows clockwise). */
    private fun angle(e: MotionEvent) = atan2(e.getY(1) - e.getY(0), e.getX(1) - e.getX(0))

    private fun wrap(deg: Float): Float {
        var d = deg % 360f
        if (d > 180f) d -= 360f
        if (d < -180f) d += 360f
        return d
    }

    private fun put(index: Int, spec: StickerSpec) {
        val l = look
        if (index !in l.stickers.indices) return
        look = l.copy(stickers = l.stickers.mapIndexed { i, s -> if (i == index) spec else s })
    }

    /** A view point in frame pixels, y up (the screen shows the middle of the frame, see [crop]). */
    private fun toFrame(g: Geo, vx: Float, vy: Float): Pair<Float, Float> {
        val (cx, cy) = crop(g.frameW, g.frameH, viewW, viewH)
        return Pair(
            (vx / viewW - 0.5f) * cx * g.frameW + g.frameW / 2f,
            ((1f - vy / viewH) - 0.5f) * cy * g.frameH + g.frameH / 2f,
        )
    }

    /** [s] moved by a view delta [dvx], [dvy], in face widths along the face's axes. */
    private fun moveBy(s: StickerSpec, g: Geo, dvx: Float, dvy: Float): StickerSpec {
        if (viewW <= 0 || viewH <= 0) return s
        val (cx, cy) = crop(g.frameW, g.frameH, viewW, viewH)
        val fx = dvx / viewW * cx * g.frameW
        val fy = -dvy / viewH * cy * g.frameH
        val fw = g.face.halfW * 2
        val dx = (fx * g.face.axX.x + fy * g.face.axX.y) / fw
        val dy = (fx * g.face.axY.x + fy * g.face.axY.y) / fw
        return s.copy(dx = (s.dx + dx).coerceIn(-2f, 2f), dy = (s.dy + dy).coerceIn(-2f, 2f))
    }

    /** The topmost of the app's stickers under view point [vx], [vy], or -1. */
    private fun hitTest(vx: Float, vy: Float): Int {
        val g = geo ?: return -1
        if (viewW <= 0 || viewH <= 0) return -1
        val (px, py) = toFrame(g, vx, vy)
        // a little extra room so small stickers are easy to catch
        val pad = g.face.halfW * 0.08f
        for (i in g.hits.indices.reversed()) {
            val h = g.hits[i]
            val rx = px - h[0]
            val ry = py - h[1]
            val u = rx * h[2] + ry * h[3]
            val v = -rx * h[3] + ry * h[2]
            if (abs(u) <= h[4] + pad && abs(v) <= h[5] + pad) return i
        }
        return -1
    }

    private fun clampZoom(z: Float): Float {
        val i = info ?: return max(1f, z)
        return z.coerceIn(max(1f, i.minZoom), i.maxZoom)
    }

    // camera (GL thread)

    private fun apply(want: Config?) {
        if (want == current && (want == null || device != null)) return
        close()
        current = want
        if (want != null) open(want)
    }

    private fun fail(message: String) {
        current = null
        post { emit("mount_error", mapOf("message" to message)) }
    }

    @SuppressLint("MissingPermission")
    private fun open(want: Config) {
        val st = texture ?: return fail("The camera could not start")
        if (activity.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            return fail("Camera permission is not granted")
        }
        val i = try {
            describe(want)
        } catch (e: Exception) {
            null
        } ?: return fail("This device has no ${want.facing} camera")
        info = i
        st.setDefaultBufferSize(i.preview.width, i.preview.height)
        val turned = i.sensorOrientation % 180 != 0
        frameW = if (turned) i.preview.height else i.preview.width
        frameH = if (turned) i.preview.width else i.preview.height
        zoom = clampZoom(zoomProp ?: zoom)
        focus = null
        reader = analysisSize(i.id)?.let { s ->
            ImageReader.newInstance(s.width, s.height, ImageFormat.YUV_420_888, 3).apply {
                setOnImageAvailableListener({ r -> analyse(r, i) }, analysis)
            }
        }
        val gen = ++generation
        try {
            manager.openCamera(i.id, object : CameraDevice.StateCallback() {
                override fun onOpened(d: CameraDevice) {
                    if (gen != generation) { d.close(); return }
                    device = d
                    configure(d, gen, reader != null)
                }
                override fun onDisconnected(d: CameraDevice) {
                    d.close()
                    if (gen == generation) { device = null; session = null; current = null }
                }
                override fun onError(d: CameraDevice, error: Int) {
                    d.close()
                    if (gen == generation) {
                        device = null; session = null
                        fail("The camera stopped (error $error)")
                    }
                }
            }, gl)
        } catch (e: Exception) {
            fail(e.message ?: "The camera could not open")
        }
    }

    @Suppress("DEPRECATION")
    private fun configure(d: CameraDevice, gen: Int, withAnalysis: Boolean) {
        val surfaces = listOfNotNull(textureSurface, if (withAnalysis) reader?.surface else null)
        try {
            d.createCaptureSession(surfaces, object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(s: CameraCaptureSession) {
                    if (gen != generation) { s.close(); return }
                    if (!withAnalysis) { reader?.close(); reader = null }
                    session = s
                    startNs = -1
                    repeat()
                    val i = info ?: return
                    post {
                        emit("camera_ready", mapOf(
                            "width" to frameW, "height" to frameH, "fps" to i.fps,
                            "facing" to if (i.front) "front" else "back", "resolution" to i.resolution,
                            "min_zoom" to max(1f, i.minZoom).toDouble(), "max_zoom" to i.maxZoom.toDouble(),
                        ))
                    }
                }
                override fun onConfigureFailed(s: CameraCaptureSession) {
                    if (gen != generation) return
                    if (withAnalysis) configure(d, gen, false) else fail("The camera could not start this size")
                }
            }, gl)
        } catch (e: Exception) {
            if (withAnalysis) configure(d, gen, false) else fail(e.message ?: "The camera could not start")
        }
    }

    /** Sends the repeating request with the current torch, zoom, focus and stabilization. */
    private fun repeat() {
        val d = device ?: return
        val s = session ?: return
        val i = info ?: return
        try {
            s.setRepeatingRequest(request(d, i, false), null, gl)
        } catch (e: Exception) {
        }
    }

    private fun request(d: CameraDevice, i: Info, trigger: Boolean): CaptureRequest {
        val b = d.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
        textureSurface?.let { b.addTarget(it) }
        reader?.let { b.addTarget(it.surface) }
        i.fpsRange?.let { b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it) }
        b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        val stab = stabilization
        if (Build.VERSION.SDK_INT >= 33 && i.previewStab) {
            b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                if (stab) CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_PREVIEW_STABILIZATION
                else CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
        } else if (i.videoStab) {
            b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                if (stab) CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_ON else CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
        }
        if (i.ois) {
            b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                if (stab) CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_ON else CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)
        }
        if (i.torch) {
            b.set(CaptureRequest.FLASH_MODE, if (torch) CameraMetadata.FLASH_MODE_TORCH else CameraMetadata.FLASH_MODE_OFF)
        }
        val z = zoom
        if (Build.VERSION.SDK_INT >= 30 && i.maxZoom > 1f) {
            b.set(CaptureRequest.CONTROL_ZOOM_RATIO, z)
        } else if (z > 1f) {
            val a = i.active
            val w = (a.width() / z).toInt()
            val h = (a.height() / z).toInt()
            b.set(CaptureRequest.SCALER_CROP_REGION,
                Rect(a.centerX() - w / 2, a.centerY() - h / 2, a.centerX() + w / 2, a.centerY() + h / 2))
        }
        val f = focus
        if (f != null) {
            b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
            b.set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(f))
            b.set(CaptureRequest.CONTROL_AE_REGIONS, arrayOf(f))
            if (trigger) b.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
        } else {
            b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
        }
        return b.build()
    }

    /** Focuses and meters at a view point, then goes back to continuous focus after a few seconds. */
    private fun focusAt(x: Float, y: Float) {
        val d = device ?: return
        val s = session ?: return
        val i = info ?: return
        val vw = viewW.toFloat()
        val vh = viewH.toFloat()
        if (vw <= 0 || vh <= 0 || frameW <= 0) return
        val (cx, cy) = crop(frameW, frameH, viewW, viewH)
        // view -> upright frame (y down), unmirrored
        var u = 0.5f + (x / vw - 0.5f) * cx
        val v = 0.5f + (y / vh - 0.5f) * cy
        if (i.front) u = 1 - u
        // upright -> sensor frame
        var sx: Float
        var sy: Float
        when (i.sensorOrientation) {
            90 -> { sx = v; sy = 1 - u }
            180 -> { sx = 1 - u; sy = 1 - v }
            270 -> { sx = 1 - v; sy = u }
            else -> { sx = u; sy = v }
        }
        // the preview stream is the middle of the sensor at its own shape, then zoomed
        val a = i.active
        val arrayAspect = a.width().toFloat() / a.height()
        val previewAspect = i.preview.width.toFloat() / i.preview.height
        val kx = if (previewAspect < arrayAspect) previewAspect / arrayAspect else 1f
        val ky = if (previewAspect > arrayAspect) arrayAspect / previewAspect else 1f
        val z = max(1f, zoom)
        sx = 0.5f + (sx - 0.5f) * kx / z
        sy = 0.5f + (sy - 0.5f) * ky / z
        val half = (min(a.width(), a.height()) * 0.05f / z).toInt().coerceAtLeast(10)
        val px = (a.left + sx * a.width()).toInt()
        val py = (a.top + sy * a.height()).toInt()
        val r = Rect(px - half, py - half, px + half, py + half)
        r.intersect(a)
        focus = MeteringRectangle(r, MeteringRectangle.METERING_WEIGHT_MAX - 1)
        val token = ++focusToken
        try {
            s.setRepeatingRequest(request(d, i, false), null, gl)
            s.capture(request(d, i, true), null, gl)
        } catch (e: Exception) {
        }
        gl.postDelayed({
            if (token == focusToken) { focus = null; repeat() }
        }, 5000)
    }

    private fun close() {
        generation++
        try { session?.close() } catch (e: Exception) {}
        session = null
        try { device?.close() } catch (e: Exception) {}
        device = null
        reader?.close()
        reader = null
        current = null
        faceRaw = null
    }

    // frames (GL thread)

    private fun attachDisplay(s: Surface) {
        val e = egl ?: return
        detachDisplay()
        display = try { e.window(s) } catch (x: Exception) { null }
    }

    private fun detachDisplay() {
        display?.let { egl?.destroy(it) }
        display = null
    }

    private fun frame() {
        val e = egl ?: return
        val r = renderer ?: return
        val st = texture ?: return
        val i = info
        val screen = display
        if (screen != null) e.current(screen) else e.idle()
        try {
            st.updateTexImage()
        } catch (x: Exception) {
            return
        }
        if (i == null || session == null || frameW <= 0) return
        st.getTransformMatrix(stMatrix)
        var ts = st.timestamp
        if (i.realtime) ts -= SystemClock.elapsedRealtimeNanos() - System.nanoTime()
        if (startNs < 0) startNs = ts
        val time = ((ts - startNs) / 1_000_000L % 1_000_000L) / 1000f
        val l = look
        val f = face(l)
        val fbo = r.render(stMatrix, frameW, frameH, time, l, f)
        geo = if (f != null && r.hits.isNotEmpty()) Geo(f, frameW, frameH, r.hits) else null
        if (screen != null && viewW > 0 && viewH > 0) {
            val (cx, cy) = crop(frameW, frameH, viewW, viewH)
            r.blit(fbo, viewW, viewH, GlRenderer.texMatrix(cx, cy))
            e.swap(screen)
        }
        val rec = encoding
        val enc = encoder
        if (rec != null && enc != null) {
            val pts = rec.framePts(ts)
            if (pts >= 0) {
                e.current(enc)
                val flip = i.front && !mirrorFront
                r.blit(fbo, rec.encodeWidth, rec.encodeHeight,
                    GlRenderer.texMatrix(degrees = if (rec.rotation == 90) -90f else 0f, flipX = flip))
                e.swap(enc, pts)
            }
        }
        photo?.let { p ->
            photo = null
            val bmp = r.read(fbo)
            val flipX = i.front && !mirrorFront
            work.execute { savePhoto(bmp, flipX, p) }
        }
    }

    /** The tracked face, smoothed between detections, in frame pixels; null without one or when not needed. */
    private fun face(l: Look): Face? {
        val raw = faceRaw
        val found = l.needsFace && raw != null && SystemClock.elapsedRealtimeNanos() - faceSeenNs < 400_000_000L
        if (found != faceFound) {
            faceFound = found
            post {
                if (lastFound != found) { lastFound = found; emit("face", mapOf("found" to found)) }
            }
        }
        if (!found) { faceSmooth = null; return null }
        val s = faceSmooth
        val out = if (s == null || s.size != raw!!.size) raw!!.copyOf() else {
            for (k in s.indices) s[k] += (raw[k] - s[k]) * 0.55f
            s
        }
        faceSmooth = out
        return Face.from(out, frameW, frameH)
    }

    // face tracking (analysis thread)

    private fun analyse(r: ImageReader, i: Info) {
        val image = try { r.acquireLatestImage() } catch (e: Exception) { null } ?: return
        val t = tracker
        if (t == null || !look.needsFace) { image.close(); return }
        t.process(image, i.sensorOrientation, i.front, i.preview.width.toFloat() / i.preview.height)
    }

    private fun onFace(p: FloatArray?) {
        if (p != null) faceSeenNs = SystemClock.elapsedRealtimeNanos()
        faceRaw = p ?: faceRaw.takeIf { SystemClock.elapsedRealtimeNanos() - faceSeenNs < 400_000_000L }
    }

    // photo

    fun takePhoto(quality: Float, reply: PxReply) {
        gl.post {
            if (session == null) post { reply(null, "The camera is not running") }
            else photo = Photo(quality, reply)
        }
    }

    private fun savePhoto(gl: Bitmap, flipX: Boolean, p: Photo) {
        try {
            val m = android.graphics.Matrix().apply { postScale(if (flipX) -1f else 1f, -1f) }
            val bmp = Bitmap.createBitmap(gl, 0, 0, gl.width, gl.height, m, false)
            gl.recycle()
            val dir = File(activity.cacheDir, "FilterCamera").apply { mkdirs() }
            val f = File(dir, "${java.util.UUID.randomUUID()}.jpg")
            f.outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, (p.quality * 100).toInt().coerceIn(1, 100), it) }
            val out = mapOf("uri" to Uri.fromFile(f).toString(), "width" to bmp.width, "height" to bmp.height)
            bmp.recycle()
            post { p.reply(out, null) }
        } catch (e: Exception) {
            post { p.reply(null, e.message ?: "The photo could not be saved") }
        }
    }

    // recording (main thread)

    fun startRecording(args: JSONObject, reply: PxReply) {
        if (recorder != null || stopping) return reply(null, "A recording is already running")
        val mute = args.optBoolean("mute", false)
        val speed = args.optDouble("speed", 1.0).toFloat()
        val audio = !mute && speed == 1f
        if (audio && activity.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return reply(null, "Microphone permission is not granted")
        }
        val segment = args.optDouble("segmentSeconds", 30.0)
        val maxDuration = if (args.has("maxDuration")) args.getDouble("maxDuration") else null
        stopping = true
        gl.post {
            val i = info
            val w = frameW
            val h = frameH
            if (session == null || i == null || w <= 0) {
                post { stopping = false; reply(null, "The camera is not running") }
                return@post
            }
            work.execute {
                val rec = try {
                    VideoRecorder(activity, w, h, i.fps, audio, speed, segment, maxDuration, object : VideoRecorder.Listener {
                        override fun onProgress(progress: Map<String, Any>) = post { emit("recording_progress", progress) }
                        override fun onAutoStop(reason: String) = post { stop(reason, null) }
                    })
                } catch (e: Exception) {
                    post { stopping = false; reply(null, e.message ?: "The recording could not start") }
                    return@execute
                }
                gl.post {
                    val e = egl
                    val surface = try { e?.window(rec.surface) } catch (x: Exception) { null }
                    if (e == null || surface == null || session == null) {
                        work.execute { rec.abandon() }
                        post { stopping = false; reply(null, "The recording could not start") }
                        return@post
                    }
                    encoder = surface
                    encoding = rec
                    rec.start()
                    post {
                        stopping = false
                        if (disposed) { stop("unmounted", null); return@post }
                        recorder = rec
                        activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                        reply(null, null)
                        if (!active || !foreground) sync()
                    }
                }
            }
        }
    }

    fun pauseRecording() { recorder?.pause() }

    fun resumeRecording() { recorder?.resume() }

    /** Stops the recording; [reply] (if any) and `recording_finished` get the result. */
    fun stop(reason: String, reply: PxReply?) {
        val rec = recorder
        if (rec == null) {
            if (reply != null) { if (stopping) stopReplies.add(reply) else reply(null, null) }
            return
        }
        reply?.let { stopReplies.add(it) }
        recorder = null
        stopping = true
        gl.post {
            encoder?.let { egl?.destroy(it) }
            encoder = null
            encoding = null
            work.execute {
                var error: String? = null
                val result = try {
                    rec.finish(reason)
                } catch (e: Exception) {
                    error = e.message ?: "The recording could not be saved"
                    null
                }
                post {
                    stopping = false
                    activity.window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    if (result != null) emit("recording_finished", result)
                    val replies = stopReplies.toList()
                    stopReplies.clear()
                    replies.forEach { it(result, error) }
                    sync()
                }
            }
        }
    }

    // teardown

    override fun dispose() {
        if (disposed) return
        activity.application.unregisterActivityLifecycleCallbacks(callbacks)
        if (Build.VERSION.SDK_INT >= 29) {
            (thermal as? PowerManager.OnThermalStatusChangedListener)?.let {
                (activity.getSystemService(Context.POWER_SERVICE) as PowerManager).removeThermalStatusListener(it)
            }
        }
        if (recorder != null) stop("unmounted", null)
        disposed = true
        gl.post {
            close()
            tracker?.close()
            tracker = null
            photo?.let { p -> post { p.reply(null, "The camera view was removed") } }
            photo = null
            // a recording being stopped still needs the GL thread for its last steps; they are posted before this
            renderer?.release()
            renderer = null
            texture?.release()
            texture = null
            textureSurface?.release()
            textureSurface = null
            detachDisplay()
            egl?.release()
            egl = null
            glThread.quitSafely()
            analysisThread.quitSafely()
        }
        work.execute { }
        work.shutdown()
    }

    private fun post(block: () -> Unit) {
        main.post(block)
    }

    // camera choice

    private fun describe(want: Config): Info? {
        val id = cameraId(manager, want.facing == "front") ?: return null
        val c = manager.getCameraCharacteristics(id)
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return null
        val sizes = map.getOutputSizes(SurfaceTexture::class.java) ?: return null
        val target = TARGETS[want.resolution] ?: TARGETS.getValue("1080p")
        val size = pickSize(sizes, target) ?: return null
        val ranges = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES).orEmpty()
        fun canDo(f: Int): Boolean {
            val minNs = try { map.getOutputMinFrameDuration(SurfaceTexture::class.java, size) } catch (e: Exception) { 0L }
            return ranges.any { it.upper >= f } && (minNs <= 0 || minNs <= 1_000_000_000L / f * 21 / 20)
        }
        val f = if (want.fps >= 60 && canDo(60)) 60 else 30
        val range = ranges.filter { it.upper == f }.maxByOrNull { it.lower } ?: ranges.filter { it.upper >= f }.minByOrNull { it.upper }
        val modes = c.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES) ?: IntArray(0)
        val ois = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION) ?: IntArray(0)
        val active = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: Rect(0, 0, size.width, size.height)
        var minZoom = 1f
        var maxZoom = c.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f
        if (Build.VERSION.SDK_INT >= 30) {
            c.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)?.let { minZoom = it.lower; maxZoom = it.upper }
        }
        return Info(
            id = id,
            front = c.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT,
            sensorOrientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90,
            realtime = c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME,
            preview = size, fps = f, fpsRange = range, resolution = resolutionName(size),
            active = active, minZoom = minZoom, maxZoom = max(1f, maxZoom),
            torch = c.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true,
            videoStab = CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_ON in modes,
            previewStab = Build.VERSION.SDK_INT >= 33 && CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_PREVIEW_STABILIZATION in modes,
            ois = CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_ON in ois,
        )
    }

    private fun analysisSize(id: String): Size? {
        val map = manager.getCameraCharacteristics(id).get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return null
        val sizes = map.getOutputSizes(ImageFormat.YUV_420_888) ?: return null
        return sizes.firstOrNull { it.width == 640 && it.height == 360 }
            ?: sizes.firstOrNull { it.width == 640 && it.height == 480 }
            ?: sizes.filter { it.width in 320..800 }.minByOrNull { abs(it.width - 640) }
    }

    companion object {
        val TARGETS = linkedMapOf("720p" to Size(1280, 720), "1080p" to Size(1920, 1080), "4k" to Size(3840, 2160))

        fun cameraId(m: CameraManager, front: Boolean): String? {
            val want = if (front) CameraCharacteristics.LENS_FACING_FRONT else CameraCharacteristics.LENS_FACING_BACK
            return m.cameraIdList.firstOrNull { m.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == want }
        }

        /** The asked 16:9 size, else the biggest 16:9 size under it, else the biggest size under it. */
        fun pickSize(sizes: Array<Size>, target: Size): Size? {
            sizes.firstOrNull { it.width == target.width && it.height == target.height }?.let { return it }
            val under = sizes.filter { it.width <= target.width && it.height <= target.height }
            val wide = under.filter { abs(it.width * 9 - it.height * 16) <= it.width / 50 }
            return (wide.ifEmpty { under }).maxByOrNull { it.width.toLong() * it.height }
                ?: sizes.minByOrNull { it.width.toLong() * it.height }
        }

        fun resolutionName(s: Size) = when {
            s.width >= 3840 -> "4k"
            s.width >= 1920 -> "1080p"
            else -> "720p"
        }

        /** The scale that crops a [fw]x[fh] frame to fill a [vw]x[vh] view. */
        fun crop(fw: Int, fh: Int, vw: Int, vh: Int): Pair<Float, Float> {
            val fa = fw.toFloat() / fh
            val va = vw.toFloat() / vh
            return if (fa > va) Pair(va / fa, 1f) else Pair(1f, fa / va)
        }

        fun thermalName(s: Int) = when (s) {
            0 -> "none"
            1 -> "light"
            2 -> "moderate"
            3 -> "severe"
            4 -> "critical"
            5 -> "emergency"
            else -> "shutdown"
        }
    }
}
