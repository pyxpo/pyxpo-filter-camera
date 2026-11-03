package dev.pyxpo.modules.filtercamera

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.MediaCodecList
import android.os.Build
import android.util.Size
import dev.pyxpo.moduleapi.PxModule
import dev.pyxpo.moduleapi.PxModuleContext
import dev.pyxpo.moduleapi.PxNativeView
import dev.pyxpo.moduleapi.PxReply
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors

/**
 * The filter camera on Android: a Camera2 + OpenGL view (`PxFilterCameraView`) with color filters, beauty, face
 * lenses and effects drawn into both the preview and the recorded video, plus crash-safe segmented recording.
 */
class FilterCameraModule : PxModule() {
    override val name = "FilterCamera"
    override val viewTypes = setOf("PxFilterCameraView")

    private val work = Executors.newSingleThreadExecutor()

    override fun makeView(ctx: PxModuleContext, type: String, emit: (name: String, value: Any?) -> Unit): PxNativeView =
        FilterCameraView(ctx.activity, emit)

    override fun call(ctx: PxModuleContext, method: String, args: JSONArray, reply: PxReply) {
        val a = ctx.activity
        try {
            when (method) {
                "getPermission" -> reply(status(a, perm(args.getString(0))), null)
                "requestPermission" -> request(ctx, perm(args.getString(0)), reply)
                "capabilities" -> work.execute { answer(a, reply) { capabilities(a) } }
                "recoverRecordings" -> work.execute { answer(a, reply) { VideoRecorder.recover(a) } }
                "takePhoto" -> {
                    val v = view(ctx, args) ?: return reply(null, UNMOUNTED)
                    val o = args.optJSONObject(1) ?: JSONObject()
                    v.takePhoto(o.optDouble("quality", 0.9).toFloat(), reply)
                }
                "startRecording" -> {
                    val v = view(ctx, args) ?: return reply(null, UNMOUNTED)
                    v.startRecording(args.optJSONObject(1) ?: JSONObject(), reply)
                }
                "pauseRecording" -> {
                    val v = view(ctx, args) ?: return reply(null, UNMOUNTED)
                    v.pauseRecording()
                    reply(null, null)
                }
                "resumeRecording" -> {
                    val v = view(ctx, args) ?: return reply(null, UNMOUNTED)
                    v.resumeRecording()
                    reply(null, null)
                }
                "stopRecording" -> {
                    val v = view(ctx, args) ?: return reply(null, UNMOUNTED)
                    v.stop("stopped", reply)
                }
                else -> reply(null, "FilterCamera has no method $method")
            }
        } catch (e: Exception) {
            reply(null, e.message ?: e.toString())
        }
    }

    private fun answer(a: Activity, reply: PxReply, block: () -> Any?) {
        val (value, error) = try {
            Pair(block(), null)
        } catch (e: Exception) {
            Pair(null, e.message ?: e.toString())
        }
        a.runOnUiThread { reply(value, error) }
    }

    private fun view(ctx: PxModuleContext, args: JSONArray) = ctx.view(args.getInt(0)) as? FilterCameraView

    /** Per camera facing: the sizes and frame rates both the camera and the video encoder can do. */
    private fun capabilities(c: Context): List<Map<String, Any>> {
        val m = c.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val encoder = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos
            .filter { it.isEncoder && it.supportedTypes.any { t -> t.equals(VideoRecorder.AVC, true) } }
            .sortedBy { if (Build.VERSION.SDK_INT >= 29 && it.isSoftwareOnly) 1 else 0 }
            .firstOrNull()?.getCapabilitiesForType(VideoRecorder.AVC)?.videoCapabilities
        fun encodes(s: Size, fps: Int): Boolean {
            val v = encoder ?: return false
            return listOf(s.width to s.height, s.height to s.width).any { (w, h) ->
                try {
                    v.isSizeSupported(w, h) && v.getSupportedFrameRatesFor(w, h).upper >= fps * 0.9
                } catch (e: Exception) {
                    false
                }
            }
        }
        return listOf("back", "front").mapNotNull { facing ->
            val id = FilterCameraView.cameraId(m, facing == "front") ?: return@mapNotNull null
            val ch = m.getCameraCharacteristics(id)
            val map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return@mapNotNull null
            val sizes = map.getOutputSizes(SurfaceTexture::class.java) ?: return@mapNotNull null
            val ranges = ch.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES).orEmpty()
            val resolutions = LinkedHashMap<String, List<Int>>()
            for ((name, target) in FilterCameraView.TARGETS) {
                val s = sizes.firstOrNull { it.width == target.width && it.height == target.height } ?: continue
                val minNs = try { map.getOutputMinFrameDuration(SurfaceTexture::class.java, s) } catch (e: Exception) { 0L }
                val rates = listOf(30, 60).filter { f ->
                    ranges.any { it.upper >= f } && (minNs <= 0 || minNs <= 1_000_000_000L / f * 21 / 20) && encodes(s, f)
                }
                if (rates.isNotEmpty()) resolutions[name] = rates
            }
            var maxZoom = ch.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f
            if (Build.VERSION.SDK_INT >= 30) ch.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)?.let { maxZoom = it.upper }
            mapOf(
                "facing" to facing,
                "resolutions" to resolutions,
                "max_zoom" to maxZoom.toDouble(),
                "torch" to (ch.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true),
            )
        }
    }

    private fun perm(kind: String) = if (kind == "microphone") Manifest.permission.RECORD_AUDIO else Manifest.permission.CAMERA

    private fun prefs(a: Activity) = a.getSharedPreferences("pyxpo.permissions", Context.MODE_PRIVATE)

    private fun status(a: Activity, p: String): Map<String, Any> = when {
        a.checkSelfPermission(p) == PackageManager.PERMISSION_GRANTED -> mapOf("status" to "granted", "canAskAgain" to true)
        !prefs(a).getBoolean(p, false) -> mapOf("status" to "undetermined", "canAskAgain" to true)
        else -> mapOf("status" to "denied", "canAskAgain" to a.shouldShowRequestPermissionRationale(p))
    }

    private fun request(ctx: PxModuleContext, p: String, reply: PxReply) {
        val a = ctx.activity
        val now = status(a, p)
        if (now["status"] == "granted" || now["canAskAgain"] != true) {
            reply(now, null)
            return
        }
        ctx.requestPermissions(arrayOf(p)) { _ ->
            prefs(a).edit().putBoolean(p, true).apply()
            reply(status(a, p), null)
        }
    }

    companion object {
        const val UNMOUNTED = "The FilterCameraView is not mounted"
    }
}
