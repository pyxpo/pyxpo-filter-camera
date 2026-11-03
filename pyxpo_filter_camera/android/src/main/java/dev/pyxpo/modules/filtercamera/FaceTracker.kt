package dev.pyxpo.modules.filtercamera

import android.media.Image
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Finds the main face in small camera frames with ML Kit and reports it in preview frame units (0..1, y down, the
 * front camera mirrored like the preview). One frame at a time: frames that come while one is checked are dropped.
 *
 * The result layout is: screen eyes (2 points), nose tip, mouth, then the face outline from [CONTOUR] on.
 */
internal class FaceTracker(private val onFace: (FloatArray?) -> Unit) {
    private val detector: FaceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .build()
    )
    private val busy = AtomicBoolean(false)
    @Volatile private var closed = false

    /**
     * Checks [image] (YUV, [sensorOrientation] from the camera). [previewAspect] is the preview's sensor-space
     * width / height, so frames of another shape are mapped onto the part the preview shows.
     */
    fun process(image: Image, sensorOrientation: Int, front: Boolean, previewAspect: Float) {
        if (closed || !busy.compareAndSet(false, true)) { image.close(); return }
        val w = image.width
        val h = image.height
        val input = try {
            InputImage.fromMediaImage(image, sensorOrientation)
        } catch (e: Exception) {
            image.close(); busy.set(false); return
        }
        detector.process(input)
            .addOnSuccessListener(DIRECT) { faces ->
                val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                val out = face?.let { layout(it, w, h, sensorOrientation, front, previewAspect) }
                if (!closed) onFace(out)
            }
            .addOnFailureListener(DIRECT) { if (!closed) onFace(null) }
            .addOnCompleteListener(DIRECT) { image.close(); busy.set(false) }
    }

    fun close() {
        closed = true
        detector.close()
    }

    private fun layout(face: com.google.mlkit.vision.face.Face, w: Int, h: Int, rot: Int, front: Boolean, previewAspect: Float): FloatArray? {
        val outline = face.getContour(FaceContour.FACE)?.points ?: return null
        val leftEye = face.getContour(FaceContour.LEFT_EYE)?.points ?: return null
        val rightEye = face.getContour(FaceContour.RIGHT_EYE)?.points ?: return null
        if (outline.size < 8 || leftEye.isEmpty() || rightEye.isEmpty()) return null
        val noseTip = face.getContour(FaceContour.NOSE_BRIDGE)?.points?.lastOrNull()
            ?: face.getContour(FaceContour.NOSE_BOTTOM)?.points?.let { it.getOrNull(it.size / 2) }
            ?: return null
        val lips = (face.getContour(FaceContour.UPPER_LIP_TOP)?.points.orEmpty() +
            face.getContour(FaceContour.LOWER_LIP_BOTTOM)?.points.orEmpty())
        // ML Kit's points are in the upright frame: the sensor frame turned by the sensor orientation
        val uw = if (rot % 180 == 0) w else h
        val uh = if (rot % 180 == 0) h else w
        val mapper = Mapper(rot, front, w.toFloat() / h, previewAspect)
        val out = FloatArray(CONTOUR + outline.size * 2)
        fun put(i: Int, x: Float, y: Float) { mapper.map(x / uw, y / uh, out, i) }
        fun centroid(pts: List<android.graphics.PointF>, i: Int) =
            put(i, pts.sumOf { it.x.toDouble() }.toFloat() / pts.size, pts.sumOf { it.y.toDouble() }.toFloat() / pts.size)
        centroid(leftEye, 0)
        centroid(rightEye, 2)
        put(4, noseTip.x, noseTip.y)
        if (lips.isNotEmpty()) centroid(lips, 6) else put(6, noseTip.x, noseTip.y)
        outline.forEachIndexed { k, p -> put(CONTOUR + k * 2, p.x, p.y) }
        return out
    }

    /**
     * Upright analysis-frame units to upright preview units: back to the sensor frame, onto the preview's part of
     * it, then upright again (mirrored for the front camera, as the preview is).
     */
    internal class Mapper(private val rot: Int, private val front: Boolean, analysisAspect: Float, previewAspect: Float) {
        private val kx: Float
        private val ky: Float

        init {
            // the camera crops one stream to the other's shape around the middle of the sensor
            if (analysisAspect < previewAspect) { kx = 1f; ky = analysisAspect / previewAspect }
            else { kx = previewAspect / analysisAspect; ky = 1f }
        }

        fun map(u: Float, v: Float, out: FloatArray, i: Int) {
            var sx: Float
            var sy: Float
            when (rot) {
                90 -> { sx = v; sy = 1 - u }
                180 -> { sx = 1 - u; sy = 1 - v }
                270 -> { sx = 1 - v; sy = u }
                else -> { sx = u; sy = v }
            }
            sx = (sx - (1 - kx) / 2) / kx
            sy = (sy - (1 - ky) / 2) / ky
            var x: Float
            val y: Float
            when (rot) {
                90 -> { x = 1 - sy; y = sx }
                180 -> { x = 1 - sx; y = 1 - sy }
                270 -> { x = sy; y = 1 - sx }
                else -> { x = sx; y = sy }
            }
            if (front) x = 1 - x
            out[i] = x
            out[i + 1] = y
        }
    }

    companion object {
        const val CONTOUR = 8
        private val DIRECT = Executor { it.run() }
    }
}
