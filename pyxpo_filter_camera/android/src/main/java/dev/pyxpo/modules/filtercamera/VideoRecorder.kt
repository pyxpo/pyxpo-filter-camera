package dev.pyxpo.modules.filtercamera

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Records the filtered frames (drawn by the GL thread into [surface]) and the microphone into numbered mp4 segments,
 * then joins them. Only finished segments are listed in the session file, so a crash loses at most the segment
 * being written; [recover] joins what a crashed session left.
 *
 * Times: the GL thread passes each frame's camera time (System.nanoTime base) to [framePts], which leaves out
 * paused time, applies the speed, and returns the video time to stamp, or -1 to skip the frame.
 */
internal class VideoRecorder(
    private val context: Context,
    /** The frame size (portrait). */
    val width: Int,
    val height: Int,
    private val fps: Int,
    private val audio: Boolean,
    private val speed: Float,
    segmentSeconds: Double,
    maxDurationSeconds: Double?,
    private val listener: Listener,
) {
    interface Listener {
        fun onProgress(progress: Map<String, Any>)
        fun onAutoStop(reason: String)
    }

    private val id = UUID.randomUUID().toString()
    private val dir = File(sessionsDir(context), id)
    private val plan = plan(width, height, fps)
    /** The encoded size; the video is stored turned when [rotation] is 90 (players turn it back). */
    val encodeWidth get() = plan.width
    val encodeHeight get() = plan.height
    val rotation get() = plan.rotation

    private val thread = HandlerThread("FilterCamera.encoder").apply { start() }
    private val handler = Handler(thread.looper)
    private val segments: Segments
    private val video: MediaCodec
    private var inputSurface: Surface? = null
    val surface: Surface get() = inputSurface!!
    private val videoEos = CountDownLatch(1)

    private var audioRecord: AudioRecord? = null
    private var audioCodec: MediaCodec? = null
    private var audioThread: Thread? = null
    @Volatile private var audioStop = false

    private val maxNs = maxDurationSeconds?.let { (it * 1e9).toLong() }
    private val frameNs = 1_000_000_000L / fps
    @Volatile private var startNs = -1L
    @Volatile private var pausedNs = 0L
    @Volatile private var pauseAt = -1L
    @Volatile private var lastPts = -1L
    @Volatile private var frames = 0
    @Volatile private var autoStopped = false
    @Volatile private var finished = false
    private var tickFrames = 0

    val paused get() = pauseAt >= 0

    init {
        dir.mkdirs()
        active.add(id)
        segments = Segments(dir, audio, plan.rotation, (segmentSeconds * 1e6).toLong(), width, height, frameNs / 1000)
        video = MediaCodec.createByCodecName(plan.codec)
        try {
            video.setCallback(object : MediaCodec.Callback() {
                override fun onInputBufferAvailable(codec: MediaCodec, index: Int) {}
                override fun onOutputBufferAvailable(codec: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
                    try {
                        val buf = codec.getOutputBuffer(index)
                        if (buf != null && info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                            buf.position(info.offset).limit(info.offset + info.size)
                            segments.write(true, buf, info)
                        }
                        codec.releaseOutputBuffer(index, false)
                    } catch (e: Exception) {
                        fail("video output", e)
                    }
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) videoEos.countDown()
                }
                override fun onOutputFormatChanged(codec: MediaCodec, format: MediaFormat) = segments.format(true, format)
                override fun onError(codec: MediaCodec, e: MediaCodec.CodecException) {
                    videoEos.countDown()
                    fail("video encoder", e)
                }
            }, handler)
            configureVideo()
            inputSurface = video.createInputSurface()
            if (audio) startAudioCodec()
            video.start()
        } catch (e: Exception) {
            release()
            dir.deleteRecursively()
            active.remove(id)
            throw e
        }
    }

    private fun configureVideo() {
        val format = MediaFormat.createVideoFormat(AVC, plan.width, plan.height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, plan.bitrate)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            if (plan.vbr) setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
        }
        if (plan.highLevel > 0) {
            format.setInteger(MediaFormat.KEY_PROFILE, MediaCodecInfo.CodecProfileLevel.AVCProfileHigh)
            format.setInteger(MediaFormat.KEY_LEVEL, plan.highLevel)
            try {
                video.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                return
            } catch (e: Exception) {
                video.reset()
                format.removeKey(MediaFormat.KEY_PROFILE)
                format.removeKey(MediaFormat.KEY_LEVEL)
            }
        }
        video.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
    }

    @SuppressLint("MissingPermission")
    private fun startAudioCodec() {
        val min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(MediaRecorder.AudioSource.CAMCORDER, RATE, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, max(min * 2, CHUNK * 4))
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            throw IllegalStateException("The microphone is not available")
        }
        audioRecord = rec
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
        audioCodec = codec
        codec.configure(MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, RATE, 1).apply {
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_BIT_RATE, 128_000)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, CHUNK * 2)
        }, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
    }

    /** Starts the clock, the microphone and the progress ticks. */
    fun start() {
        startNs = System.nanoTime()
        audioRecord?.let { rec ->
            rec.startRecording()
            audioThread = Thread({ audioLoop(rec, audioCodec!!) }, "FilterCamera.audio").apply { start() }
        }
        handler.postDelayed(::tick, 1000)
    }

    /** The video time in ns for a frame taken at [cameraNs] (System.nanoTime base), or -1 to leave it out. */
    fun framePts(cameraNs: Long): Long {
        if (startNs < 0 || paused || finished || autoStopped) return -1
        val raw = cameraNs - startNs - pausedNs
        if (raw < 0) return -1
        val pts = if (speed == 1f) raw else (raw / speed).toLong()
        val last = lastPts
        if (last >= 0 && (pts <= last || (speed > 1f && pts - last < frameNs * 3 / 4))) return -1
        if (maxNs != null && pts >= maxNs) {
            autoStop("max_duration")
            return -1
        }
        lastPts = pts
        frames++
        return pts
    }

    fun pause() {
        if (pauseAt < 0) pauseAt = System.nanoTime()
    }

    fun resume() {
        val at = pauseAt
        if (at >= 0) {
            pausedNs += System.nanoTime() - at
            pauseAt = -1
        }
    }

    private fun fail(where: String, e: Throwable) {
        Log.w(TAG, "Recording failed in the $where", e)
        autoStop("error")
    }

    private fun autoStop(reason: String) {
        if (autoStopped || finished) return
        autoStopped = true
        listener.onAutoStop(reason)
    }

    private fun tick() {
        if (finished) return
        val n = frames
        listener.onProgress(mapOf(
            "duration" to max(0L, lastPts) / 1e9,
            "size" to segments.bytes(),
            "fps" to (n - tickFrames),
            "segments" to segments.count(),
            "paused" to paused,
        ))
        tickFrames = n
        if (dir.usableSpace < MIN_FREE) autoStop("storage")
        handler.postDelayed(::tick, 1000)
    }

    private fun audioLoop(rec: AudioRecord, codec: MediaCodec) {
        val pcm = ByteBuffer.allocateDirect(CHUNK * 2)
        val info = MediaCodec.BufferInfo()
        var last = -1L
        try {
            while (!audioStop) {
                pcm.clear()
                val n = rec.read(pcm, CHUNK * 2)
                val now = System.nanoTime()
                drainAudio(codec, info, false)
                if (n <= 0 || paused || startNs < 0) continue
                val durUs = n / 2 * 1_000_000L / RATE
                var pts = (now - startNs - pausedNs) / 1000 - durUs
                if (pts < 0) continue
                if (pts <= last) pts = last + 1
                last = pts
                var idx = codec.dequeueInputBuffer(10_000)
                while (idx < 0 && !audioStop) {
                    drainAudio(codec, info, false)
                    idx = codec.dequeueInputBuffer(10_000)
                }
                if (idx < 0) break
                val input = codec.getInputBuffer(idx)!!
                input.clear()
                pcm.position(0).limit(n)
                input.put(pcm)
                codec.queueInputBuffer(idx, 0, n, pts, 0)
            }
            var idx = codec.dequeueInputBuffer(100_000)
            var tries = 0
            while (idx < 0 && tries++ < 20) {
                drainAudio(codec, info, false)
                idx = codec.dequeueInputBuffer(100_000)
            }
            if (idx >= 0) {
                codec.queueInputBuffer(idx, 0, 0, last + 1, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                drainAudio(codec, info, true)
            }
        } catch (e: Exception) {
            // a broken microphone or audio encoder leaves the rest of the video without sound, not without video
            if (!audioStop) {
                Log.w(TAG, "The sound stopped; the video goes on without it", e)
                segments.noAudio()
            }
        }
    }

    private fun drainAudio(codec: MediaCodec, info: MediaCodec.BufferInfo, untilEos: Boolean) {
        val until = System.nanoTime() + 2_000_000_000L
        while (true) {
            val idx = codec.dequeueOutputBuffer(info, if (untilEos) 10_000 else 0)
            when {
                idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> segments.format(false, codec.outputFormat)
                idx >= 0 -> {
                    val buf = codec.getOutputBuffer(idx)
                    if (buf != null && info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                        buf.position(info.offset).limit(info.offset + info.size)
                        segments.write(false, buf, info)
                    }
                    codec.releaseOutputBuffer(idx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
                !untilEos || System.nanoTime() > until -> return
            }
        }
    }

    /**
     * Ends the recording and joins the segments. Call it off the main and GL threads, after the GL thread stopped
     * drawing into [surface]. Returns the stop result, or null when no frame was recorded.
     */
    fun finish(reason: String): Map<String, Any?>? {
        finished = true
        handler.removeCallbacksAndMessages(null)
        try {
            video.signalEndOfInputStream()
            videoEos.await(3, TimeUnit.SECONDS)
        } catch (e: Exception) {
        }
        audioStop = true
        audioThread?.join(3000)
        release()
        segments.close()
        val result = try {
            join(context, dir, reason)
        } finally {
            active.remove(id)
        }
        return result
    }

    /** Throws the recording away (the view went away mid-start). */
    fun abandon() {
        finished = true
        audioStop = true
        handler.removeCallbacksAndMessages(null)
        audioThread?.join(1000)
        release()
        segments.close()
        active.remove(id)
    }

    private fun release() {
        try { video.stop() } catch (e: Exception) {}
        video.release()
        try { audioRecord?.stop() } catch (e: Exception) {}
        audioRecord?.release()
        audioRecord = null
        try { audioCodec?.stop() } catch (e: Exception) {}
        audioCodec?.release()
        audioCodec = null
        inputSurface?.release()
        inputSurface = null
        thread.quitSafely()
    }

    /** Writes samples into numbered segment files, starting a new one at the first key frame after the length. */
    private class Segments(
        val dir: File, var audio: Boolean, val rotation: Int, val segmentUs: Long,
        val width: Int, val height: Int, val frameUs: Long,
    ) {
        private class Pending(val video: Boolean, val data: ByteArray, val pts: Long, val flags: Int)

        private var videoFormat: MediaFormat? = null
        private var audioFormat: MediaFormat? = null
        private val pending = ArrayList<Pending>()
        private var muxer: MediaMuxer? = null
        private var file: File? = null
        private var vTrack = -1
        private var aTrack = -1
        private var index = 0
        private var v0 = -1L
        private var a0 = -1L
        private var vLast = -1L
        private var aLast = -1L
        private var samples = 0
        private val done = JSONArray()
        private var doneBytes = 0L
        private var closed = false

        init {
            save()
        }

        private val ready get() = videoFormat != null && (!audio || audioFormat != null)

        @Synchronized fun format(video: Boolean, f: MediaFormat) {
            if (video) videoFormat = f else audioFormat = f
            flush()
        }

        /** The sound failed: segments no longer wait for its format (a started segment keeps its silent track). */
        @Synchronized fun noAudio() {
            audio = false
            flush()
        }

        private fun flush() {
            if (ready && pending.isNotEmpty()) {
                val info = MediaCodec.BufferInfo()
                for (p in pending) {
                    info.set(0, p.data.size, p.pts, p.flags)
                    put(p.video, ByteBuffer.wrap(p.data), info)
                }
                pending.clear()
            }
        }

        @Synchronized fun write(video: Boolean, buf: ByteBuffer, info: MediaCodec.BufferInfo) {
            if (closed) return
            if (!ready) {
                if (pending.size < 600) {
                    val data = ByteArray(info.size)
                    buf.get(data)
                    pending.add(Pending(video, data, info.presentationTimeUs, info.flags))
                }
                return
            }
            put(video, buf, info)
        }

        private fun put(video: Boolean, buf: ByteBuffer, info: MediaCodec.BufferInfo) {
            val key = info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME != 0
            if (video && key && muxer != null && v0 >= 0 && info.presentationTimeUs - v0 >= segmentUs) finishSegment()
            if (muxer == null) {
                // a segment starts with a video key frame
                if (!video || !key) return
                open()
            }
            val m = muxer ?: return
            if (video) {
                if (info.presentationTimeUs <= vLast) return
                m.writeSampleData(vTrack, buf, info)
                if (v0 < 0) v0 = info.presentationTimeUs
                vLast = info.presentationTimeUs
            } else {
                if (aTrack < 0 || info.presentationTimeUs <= aLast) return
                m.writeSampleData(aTrack, buf, info)
                if (a0 < 0) a0 = info.presentationTimeUs
                aLast = info.presentationTimeUs
            }
            samples++
        }

        private fun open() {
            val f = File(dir, "seg_%04d.mp4".format(index++))
            val m = MediaMuxer(f.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            m.setOrientationHint(rotation)
            vTrack = m.addTrack(videoFormat!!)
            aTrack = audioFormat?.takeIf { audio }?.let { m.addTrack(it) } ?: -1
            m.start()
            muxer = m
            file = f
            v0 = -1; a0 = -1; vLast = -1; aLast = -1; samples = 0
        }

        private fun finishSegment() {
            val m = muxer ?: return
            val f = file!!
            muxer = null
            file = null
            val ok = try {
                m.stop()
                v0 >= 0
            } catch (e: Exception) {
                false
            } finally {
                m.release()
            }
            if (!ok) { f.delete(); return }
            done.put(JSONObject().put("file", f.name).put("v0", v0).put("a0", a0).put("end", vLast + frameUs))
            doneBytes += f.length()
            save()
        }

        private fun save() {
            val meta = JSONObject().put("version", 1).put("width", width).put("height", height)
                .put("rotation", rotation).put("segments", done)
            val tmp = File(dir, "session.json.tmp")
            tmp.writeText(meta.toString())
            tmp.renameTo(File(dir, SESSION))
        }

        @Synchronized fun bytes(): Long = doneBytes + (file?.length() ?: 0L)
        @Synchronized fun count(): Int = done.length() + if (muxer != null) 1 else 0

        @Synchronized fun close() {
            if (closed) return
            finishSegment()
            closed = true
        }
    }

    private class Plan(val codec: String, val width: Int, val height: Int, val rotation: Int, val bitrate: Int,
                       val vbr: Boolean, val highLevel: Int)

    companion object {
        const val AVC = MediaFormat.MIMETYPE_VIDEO_AVC
        private const val TAG = "FilterCamera"
        private const val RATE = 48_000
        private const val CHUNK = 2048
        private const val SESSION = "session.json"
        private const val MIN_FREE = 200L * 1024 * 1024
        /** Sessions recording now; [recover] leaves them alone. */
        private val active: MutableSet<String> = ConcurrentHashMap.newKeySet()

        fun sessionsDir(c: Context) = File(c.filesDir, "FilterCamera/sessions")

        private fun even16(v: Double) = max(16, (v / 16).roundToInt() * 16)

        /** The encoder, size and bitrate for a [w]x[h] frame at [fps]: as asked, or turned, or smaller. */
        private fun plan(w: Int, h: Int, fps: Int): Plan {
            val info = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos
                .filter { it.isEncoder && it.supportedTypes.any { t -> t.equals(AVC, true) } }
                .sortedBy { if (Build.VERSION.SDK_INT >= 29 && it.isSoftwareOnly) 1 else 0 }
                .firstOrNull() ?: throw IllegalStateException("This device has no H.264 encoder")
            val caps = info.getCapabilitiesForType(AVC)
            val vc = caps.videoCapabilities ?: throw IllegalStateException("The H.264 encoder has no video settings")
            fun ok(a: Int, b: Int) = try {
                vc.isSizeSupported(a, b) && vc.getSupportedFrameRatesFor(a, b).upper >= fps * 0.9
            } catch (e: Exception) {
                false
            }
            var size: IntArray? = null
            var s = 1.0
            for (i in 0 until 8) {
                val a = if (i == 0) w else even16(w * s)
                val b = if (i == 0) h else even16(h * s)
                if (ok(a, b)) { size = intArrayOf(a, b, 0); break }
                if (ok(b, a)) { size = intArrayOf(b, a, 90); break }
                s *= 0.75
            }
            val (ew, eh, rot) = size ?: intArrayOf(w, h, 0)
            val pixels = ew.toLong() * eh
            var mbps = when {
                pixels >= 3840L * 2160 * 9 / 10 -> 40.0
                pixels >= 1920L * 1080 * 9 / 10 -> 12.0
                else -> 6.0
            }
            if (fps >= 50) mbps *= 1.5
            val bitrate = vc.bitrateRange.clamp((mbps * 1_000_000).toInt())
            val vbr = caps.encoderCapabilities?.isBitrateModeSupported(MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR) == true
            val high = caps.profileLevels.filter { it.profile == MediaCodecInfo.CodecProfileLevel.AVCProfileHigh }
                .maxOfOrNull { it.level } ?: 0
            return Plan(info.name, ew, eh, rot, bitrate, vbr, high)
        }

        private fun outputFile(c: Context): File {
            val d = File(c.cacheDir, "FilterCamera").apply { mkdirs() }
            return File(d, "${UUID.randomUUID()}.mp4")
        }

        /** Joins a session's finished segments into one mp4 in the cache and deletes the session. */
        fun join(c: Context, dir: File, reason: String): Map<String, Any?>? {
            val meta = try { JSONObject(File(dir, SESSION).readText()) } catch (e: Exception) { null }
            val segs = meta?.optJSONArray("segments")
            val list = (0 until (segs?.length() ?: 0)).map { segs!!.getJSONObject(it) }
                .filter { File(dir, it.getString("file")).exists() }
            if (meta == null || list.isEmpty()) {
                dir.deleteRecursively()
                return null
            }
            val out = outputFile(c)
            try {
                if (list.size == 1) {
                    val f = File(dir, list[0].getString("file"))
                    if (!f.renameTo(out)) f.copyTo(out, overwrite = true)
                } else {
                    concat(list.map { File(dir, it.getString("file")) to it }, out, meta.optInt("rotation"))
                }
            } catch (e: Exception) {
                out.delete()
                throw e
            }
            dir.deleteRecursively()
            val durationUs = list.last().getLong("end") - list.first().getLong("v0")
            return mapOf(
                "uri" to Uri.fromFile(out).toString(),
                "duration" to durationUs / 1e6,
                "size" to out.length(),
                "width" to meta.optInt("width"),
                "height" to meta.optInt("height"),
                "segments" to list.size,
                "reason" to reason,
            )
        }

        private fun concat(parts: List<Pair<File, JSONObject>>, out: File, rotation: Int) {
            val muxer = MediaMuxer(out.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer.setOrientationHint(rotation)
            var vOut = -1
            var aOut = -1
            var started = false
            var buf = ByteBuffer.allocateDirect(4 * 1024 * 1024)
            val info = MediaCodec.BufferInfo()
            val lastOut = longArrayOf(-1, -1)
            try {
                for ((file, seg) in parts) {
                    val ex = arrayOf(MediaExtractor(), MediaExtractor())
                    try {
                        ex.forEach { it.setDataSource(file.path) }
                        val fmt = arrayOfNulls<MediaFormat>(2)
                        for (t in 0 until ex[0].trackCount) {
                            val f = ex[0].getTrackFormat(t)
                            val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
                            val k = if (mime.startsWith("video/")) 0 else if (mime.startsWith("audio/")) 1 else continue
                            if (fmt[k] == null) { fmt[k] = f; ex[k].selectTrack(t) }
                        }
                        if (fmt[0] == null) continue
                        if (!started) {
                            vOut = muxer.addTrack(fmt[0]!!)
                            if (fmt[1] != null) aOut = muxer.addTrack(fmt[1]!!)
                            muxer.start()
                            started = true
                        }
                        for (k in 0..1) fmt[k]?.let { f ->
                            if (f.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) {
                                val need = f.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE)
                                if (need > buf.capacity()) buf = ByteBuffer.allocateDirect(need)
                            }
                        }
                        val tracks = intArrayOf(vOut, if (fmt[1] != null) aOut else -1)
                        val starts = longArrayOf(seg.getLong("v0"), seg.optLong("a0", -1))
                        val base = LongArray(2) { if (fmt[it] != null) ex[it].sampleTime else -1 }
                        val live = BooleanArray(2) { fmt[it] != null && tracks[it] >= 0 && base[it] >= 0 && starts[it] >= 0 }
                        while (live[0] || live[1]) {
                            val k = when {
                                !live[1] -> 0
                                !live[0] -> 1
                                starts[0] + ex[0].sampleTime - base[0] <= starts[1] + ex[1].sampleTime - base[1] -> 0
                                else -> 1
                            }
                            val e = ex[k]
                            val size = try {
                                e.readSampleData(buf, 0)
                            } catch (x: IllegalArgumentException) {
                                buf = ByteBuffer.allocateDirect(buf.capacity() * 2)
                                continue
                            }
                            if (size < 0) { live[k] = false; continue }
                            var t = starts[k] + e.sampleTime - base[k]
                            if (t <= lastOut[k]) t = lastOut[k] + 1
                            lastOut[k] = t
                            val flags = if (e.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
                            info.set(0, size, t, flags)
                            muxer.writeSampleData(tracks[k], buf, info)
                            if (!e.advance()) live[k] = false
                        }
                    } finally {
                        ex.forEach { it.release() }
                    }
                    file.delete()
                }
                if (started) muxer.stop()
            } finally {
                muxer.release()
            }
        }

        /** Joins every session a crash or kill left behind. */
        fun recover(c: Context): List<Map<String, Any?>> {
            val dirs = sessionsDir(c).listFiles()?.filter { it.isDirectory && it.name !in active } ?: return emptyList()
            return dirs.mapNotNull { d ->
                try {
                    join(c, d, "recovered")?.plus("recovered" to true)
                } catch (e: Exception) {
                    d.deleteRecursively()
                    null
                }
            }
        }
    }
}
