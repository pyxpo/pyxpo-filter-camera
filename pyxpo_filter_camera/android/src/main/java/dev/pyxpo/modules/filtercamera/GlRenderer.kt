package dev.pyxpo.modules.filtercamera

import android.graphics.Bitmap
import android.graphics.PointF
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLExt
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLUtils
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

private const val EGL_RECORDABLE_ANDROID = 0x3142

/** One EGL context for one thread. It draws into windows and into MediaCodec input surfaces (recordable config). */
internal class Egl {
    private val display: EGLDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
    private val config: EGLConfig
    private val context: EGLContext
    private val pbuffer: EGLSurface

    init {
        val version = IntArray(2)
        check(EGL14.eglInitialize(display, version, 0, version, 1)) { "EGL could not start" }
        val attrs = intArrayOf(
            EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8, EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT, EGL_RECORDABLE_ANDROID, 1, EGL14.EGL_NONE,
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val count = IntArray(1)
        check(EGL14.eglChooseConfig(display, attrs, 0, configs, 0, 1, count, 0) && count[0] > 0) { "No recordable EGL config" }
        config = configs[0]!!
        context = EGL14.eglCreateContext(display, config, EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE), 0)
        check(context != EGL14.EGL_NO_CONTEXT) { "EGL context failed" }
        pbuffer = EGL14.eglCreatePbufferSurface(display, config, intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0)
        idle()
    }

    fun window(surface: Any): EGLSurface {
        val s = EGL14.eglCreateWindowSurface(display, config, surface, intArrayOf(EGL14.EGL_NONE), 0)
        check(s != null && s != EGL14.EGL_NO_SURFACE) { "EGL window failed: 0x${Integer.toHexString(EGL14.eglGetError())}" }
        return s
    }

    fun current(s: EGLSurface) = EGL14.eglMakeCurrent(display, s, s, context)
    fun idle() = current(pbuffer)

    /** Shows the frame; [ptsNs] stamps an encoder frame. */
    fun swap(s: EGLSurface, ptsNs: Long = -1): Boolean {
        if (ptsNs >= 0) EGLExt.eglPresentationTimeANDROID(display, s, ptsNs)
        return EGL14.eglSwapBuffers(display, s)
    }

    fun destroy(s: EGLSurface) {
        idle()
        EGL14.eglDestroySurface(display, s)
    }

    fun release() {
        EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        EGL14.eglDestroySurface(display, pbuffer)
        EGL14.eglDestroyContext(display, context)
        EGL14.eglReleaseThread()
        EGL14.eglTerminate(display)
    }
}

internal class Program(vertex: String, fragment: String) {
    val id: Int = GLES20.glCreateProgram()
    val aPos: Int
    val aTex: Int
    private val locations = HashMap<String, Int>()

    init {
        val vs = compile(GLES20.GL_VERTEX_SHADER, vertex)
        val fs = compile(GLES20.GL_FRAGMENT_SHADER, fragment)
        GLES20.glAttachShader(id, vs)
        GLES20.glAttachShader(id, fs)
        GLES20.glLinkProgram(id)
        val ok = IntArray(1)
        GLES20.glGetProgramiv(id, GLES20.GL_LINK_STATUS, ok, 0)
        GLES20.glDeleteShader(vs)
        GLES20.glDeleteShader(fs)
        if (ok[0] == 0) {
            val log = GLES20.glGetProgramInfoLog(id)
            GLES20.glDeleteProgram(id)
            throw IllegalStateException("Shader link failed: $log")
        }
        aPos = GLES20.glGetAttribLocation(id, "aPos")
        aTex = GLES20.glGetAttribLocation(id, "aTex")
    }

    fun u(name: String) = locations.getOrPut(name) { GLES20.glGetUniformLocation(id, name) }
    fun f(name: String, v: Float) = u(name).let { if (it >= 0) GLES20.glUniform1f(it, v) }
    fun v2(name: String, x: Float, y: Float) = u(name).let { if (it >= 0) GLES20.glUniform2f(it, x, y) }
    fun v3(name: String, v: FloatArray) = u(name).let { if (it >= 0) GLES20.glUniform3f(it, v[0], v[1], v[2]) }
    fun v4(name: String, x: Float, y: Float, z: Float, w: Float) =
        u(name).let { if (it >= 0) GLES20.glUniform4f(it, x, y, z, w) }
    fun v3a(name: String, v: FloatArray) = u(name).let { if (it >= 0) GLES20.glUniform3fv(it, v.size / 3, v, 0) }

    companion object {
        private fun compile(type: Int, src: String): Int {
            val s = GLES20.glCreateShader(type)
            GLES20.glShaderSource(s, src)
            GLES20.glCompileShader(s)
            val ok = IntArray(1)
            GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, ok, 0)
            if (ok[0] == 0) {
                val log = GLES20.glGetShaderInfoLog(s)
                GLES20.glDeleteShader(s)
                throw IllegalStateException("Shader compile failed: $log")
            }
            return s
        }
    }
}

/** A texture to render into. */
internal class Fbo(val width: Int, val height: Int) {
    val tex: Int
    private val fb: Int

    init {
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        tex = ids[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
        texParams(GLES20.GL_TEXTURE_2D)
        GLES20.glGenFramebuffers(1, ids, 0)
        fb = ids[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fb)
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, tex, 0)
        val status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        check(status == GLES20.GL_FRAMEBUFFER_COMPLETE) { "Framebuffer ${width}x$height incomplete: $status" }
    }

    fun bind() {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fb)
        GLES20.glViewport(0, 0, width, height)
    }

    fun release() {
        GLES20.glDeleteFramebuffers(1, intArrayOf(fb), 0)
        GLES20.glDeleteTextures(1, intArrayOf(tex), 0)
    }
}

internal fun texParams(target: Int) {
    GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
    GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
    GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
    GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
}

/** What the frame looks like: the view's filter props. */
internal data class Look(
    val grade: Grade? = null, val filterIntensity: Float = 1f,
    val smooth: Float = 0f, val brighten: Float = 0f, val slim: Float = 0f, val eyes: Float = 0f,
    val lens: String = "none", val stickers: List<StickerSpec> = emptyList(), val effect: String = "none", val effectIntensity: Float = 1f,
) {
    val beauty get() = smooth > 0f || brighten > 0f || slim > 0f || eyes > 0f
    val needsFace get() = lens != "none" || stickers.isNotEmpty() || slim > 0f || eyes > 0f
}

/** A face in frame pixels, y up. [axX] runs from the screen-left eye to the other; [axY] points to the forehead. */
internal class Face(
    val eyeA: PointF, val eyeB: PointF, val nose: PointF, val mouth: PointF,
    val center: PointF, val axX: PointF, val axY: PointF, val halfW: Float, val halfH: Float,
) {
    fun anchor(a: Anchor): PointF = when (a) {
        Anchor.EYES -> PointF((eyeA.x + eyeB.x) / 2, (eyeA.y + eyeB.y) / 2)
        Anchor.NOSE -> nose
        Anchor.MOUTH -> mouth
        Anchor.FOREHEAD -> PointF(center.x + axY.x * halfH, center.y + axY.y * halfH)
    }

    companion object {
        /** [p] is [FaceTracker]'s layout in 0..1 frame units, y down. */
        fun from(p: FloatArray, w: Int, h: Int): Face {
            fun pt(i: Int) = PointF(p[i] * w, (1f - p[i + 1]) * h)
            var a = pt(0)
            var b = pt(2)
            if (a.x > b.x) a = b.also { b = a }
            val d = hypot(b.x - a.x, b.y - a.y).coerceAtLeast(1f)
            val ax = PointF((b.x - a.x) / d, (b.y - a.y) / d)
            val ay = PointF(-ax.y, ax.x)
            var cx = 0f
            var cy = 0f
            val n = (p.size - FaceTracker.CONTOUR) / 2
            for (i in 0 until n) { val q = pt(FaceTracker.CONTOUR + i * 2); cx += q.x; cy += q.y }
            cx /= n; cy /= n
            var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE; var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
            for (i in 0 until n) {
                val q = pt(FaceTracker.CONTOUR + i * 2)
                val x = (q.x - cx) * ax.x + (q.y - cy) * ax.y
                val y = (q.x - cx) * ay.x + (q.y - cy) * ay.y
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
            }
            val mx = (minX + maxX) / 2
            val my = (minY + maxY) / 2
            val center = PointF(cx + ax.x * mx + ay.x * my, cy + ax.y * mx + ay.y * my)
            return Face(a, b, pt(4), pt(6), center, ax, ay, max(1f, (maxX - minX) / 2), max(1f, (maxY - minY) / 2))
        }
    }
}

/**
 * The filter chain on the GL thread: camera frame -> beauty (or a copy) -> lens stickers -> color grade -> effect,
 * through two frame-size textures. [render] returns the texture holding the result.
 */
internal class GlRenderer {
    val oesTex: Int
    private val programs = HashMap<String, Program>()
    private var a: Fbo? = null
    private var b: Fbo? = null
    private var lensName = "none"
    private var stickers = emptyList<Pair<Sticker, Int>>()
    // the app's own stickers: a texture and its height / width per emoji; placement comes from the look each frame
    private var customEmoji = emptyList<String>()
    private var custom = emptyList<Pair<Float, Int>>()

    /**
     * Where each of the app's own stickers was last drawn, in frame pixels (y up): center x, y, the sprite's x axis
     * (unit) x, y, and half width, half height. Empty when no face was drawn on.
     */
    var hits: List<FloatArray> = emptyList()
        private set
    private val spritePos = floats(8)

    init {
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        oesTex = ids[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        texParams(GLES11Ext.GL_TEXTURE_EXTERNAL_OES)
    }

    private fun program(key: String, oes: Boolean, body: () -> String) = programs.getOrPut(key + if (oes) ":oes" else "") {
        Program(Shaders.VERTEX, Shaders.fragment(body(), oes))
    }

    fun render(stMatrix: FloatArray, w: Int, h: Int, time: Float, look: Look, face: Face?): Fbo {
        if (a?.width != w || a?.height != h) {
            a?.release(); b?.release()
            a = Fbo(w, h); b = Fbo(w, h)
        }
        var cur = a!!
        var other = b!!
        val oesTarget = GLES11Ext.GL_TEXTURE_EXTERNAL_OES

        if (look.beauty) {
            draw(program("beauty", true) { Shaders.BEAUTY }, oesTarget, oesTex, stMatrix, cur, time) { p ->
                p.f("uSmooth", look.smooth); p.f("uBright", look.brighten)
                p.f("uSlim", look.slim); p.f("uEyes", look.eyes)
                p.f("uRadius", max(1.5f, min(w, h) / 216f))
                p.f("uFace", if (face != null) 1f else 0f)
                if (face != null) {
                    p.v2("uEyeA", face.eyeA.x, face.eyeA.y); p.v2("uEyeB", face.eyeB.x, face.eyeB.y)
                    p.v2("uCenter", face.center.x, face.center.y)
                    p.v2("uAxX", face.axX.x, face.axX.y); p.v2("uAxY", face.axY.x, face.axY.y)
                    p.v2("uRad", face.halfW * 1.05f, face.halfH * 1.1f)
                    p.f("uEyeRad", hypot(face.eyeB.x - face.eyeA.x, face.eyeB.y - face.eyeA.y) * 0.42f)
                }
            }
        } else {
            draw(program("copy", true) { Shaders.COPY }, oesTarget, oesTex, stMatrix, cur, time)
        }

        if (look.lens != lensName) setLens(look.lens)
        val emoji = look.stickers.map { it.emoji }
        if (emoji != customEmoji) setCustom(emoji)
        hits = if (face != null && (stickers.isNotEmpty() || custom.isNotEmpty())) drawStickers(cur, face, look.stickers)
               else emptyList()

        look.grade?.let { g ->
            draw(program("color", false) { Shaders.COLOR }, GLES20.GL_TEXTURE_2D, cur.tex, IDENTITY, other, time) { p ->
                p.f("uIntensity", look.filterIntensity); p.f("uExposure", g.exposure); p.f("uContrast", g.contrast)
                p.f("uSaturation", g.saturation); p.f("uTemp", g.temperature); p.f("uTint", g.tint); p.f("uFade", g.fade)
                p.v3("uShadow", g.shadow); p.v3("uHigh", g.highlight); p.f("uSplit", g.split)
                p.f("uVignette", g.vignette); p.f("uGrain", g.grain); p.f("uMono", g.mono)
                p.f("uSepia", g.sepia); p.f("uHue", g.hue)
                p.v4("uTones", g.toneHighlights, g.toneShadows, g.whites, g.blacks)
                p.v4("uPresence", g.vibrance, g.clarity, g.sharpen, g.dehaze)
                p.f("uHslOn", if (g.hslOn) 1f else 0f)
                if (g.hslOn) p.v3a("uHsl[0]", g.hsl)
            }
            cur = other.also { other = cur }
        }

        Shaders.effect(look.effect)?.let { src ->
            draw(program("fx:${look.effect}", false) { src }, GLES20.GL_TEXTURE_2D, cur.tex, IDENTITY, other, time) { p ->
                p.f("uAmount", look.effectIntensity)
            }
            cur = other
        }
        return cur
    }

    /** Draws [tex] into the current surface's framebuffer, [w]x[h], sampling it through [texMatrix]. */
    fun blit(tex: Fbo, w: Int, h: Int, texMatrix: FloatArray) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glViewport(0, 0, w, h)
        draw(program("copy", false) { Shaders.COPY }, GLES20.GL_TEXTURE_2D, tex.tex, texMatrix, null, 0f)
    }

    /** The frame as a bitmap, upside down (GL rows start at the bottom). */
    fun read(fbo: Fbo): Bitmap {
        fbo.bind()
        val buf = ByteBuffer.allocateDirect(fbo.width * fbo.height * 4).order(ByteOrder.nativeOrder())
        GLES20.glReadPixels(0, 0, fbo.width, fbo.height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buf)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        return Bitmap.createBitmap(fbo.width, fbo.height, Bitmap.Config.ARGB_8888).apply { copyPixelsFromBuffer(buf) }
    }

    private fun texture(bitmap: Bitmap): Int {
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, ids[0])
        texParams(GLES20.GL_TEXTURE_2D)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)
        return ids[0]
    }

    private fun setLens(name: String) {
        stickers.forEach { GLES20.glDeleteTextures(1, intArrayOf(it.second), 0) }
        lensName = name
        stickers = Lenses.build(name).map { it to texture(it.bitmap) }
    }

    /** Makes textures for the app's stickers; only a new emoji list costs this, not moving or sizing them. */
    private fun setCustom(emoji: List<String>) {
        custom.forEach { GLES20.glDeleteTextures(1, intArrayOf(it.second), 0) }
        customEmoji = emoji
        custom = emoji.map { e ->
            val bmp = Lenses.emoji(e)
            (bmp.height.toFloat() / bmp.width) to texture(bmp).also { bmp.recycle() }
        }
    }

    private fun drawStickers(out: Fbo, face: Face, specs: List<StickerSpec>): List<FloatArray> {
        val p = program("copy", false) { Shaders.COPY }
        out.bind()
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        for ((s, tex) in stickers) {
            sprite(p, out, face, tex, s.bitmap.height.toFloat() / s.bitmap.width, s.anchor, s.width, s.dx, s.dy, s.rotation)
        }
        val placed = ArrayList<FloatArray>(custom.size)
        for (i in custom.indices) {
            val s = specs.getOrNull(i) ?: break
            val (aspect, tex) = custom[i]
            placed += sprite(p, out, face, tex, aspect, s.anchor, s.size, s.dx, s.dy, s.rotation)
        }
        GLES20.glDisable(GLES20.GL_BLEND)
        return placed
    }

    /** Draws one sticker [width] face widths wide; returns where (see [hits]). */
    private fun sprite(
        p: Program, out: Fbo, face: Face, tex: Int, aspect: Float, anchor: Anchor,
        width: Float, dx: Float, dy: Float, rotation: Float,
    ): FloatArray {
        val fw = face.halfW * 2
        val at = face.anchor(anchor)
        val cx = at.x + (face.axX.x * dx + face.axY.x * dy) * fw
        val cy = at.y + (face.axX.y * dx + face.axY.y * dy) * fw
        val hw = width * fw / 2
        val hh = hw * aspect
        // the sprite's own axes: the face's, turned clockwise (y is up, so by minus the angle)
        val t = Math.toRadians(-rotation.toDouble())
        val c = cos(t).toFloat()
        val sn = sin(t).toFloat()
        val ux = face.axX.x * c + face.axY.x * sn
        val uy = face.axX.y * c + face.axY.y * sn
        val vx = face.axY.x * c - face.axX.x * sn
        val vy = face.axY.y * c - face.axX.y * sn
        spritePos.position(0)
        for (i in 0 until 4) {
            val x = cx + ux * CORNER_X[i] * hw + vx * CORNER_Y[i] * hh
            val y = cy + uy * CORNER_X[i] * hw + vy * CORNER_Y[i] * hh
            spritePos.put(x / out.width * 2 - 1).put(y / out.height * 2 - 1)
        }
        spritePos.position(0)
        draw(p, GLES20.GL_TEXTURE_2D, tex, IDENTITY, null, 0f, spritePos, SPRITE_TEX)
        return floatArrayOf(cx, cy, ux, uy, hw, hh)
    }

    private fun draw(
        p: Program, target: Int, tex: Int, texMatrix: FloatArray, out: Fbo?, time: Float,
        pos: FloatBuffer = QUAD_POS, uv: FloatBuffer = QUAD_TEX, setup: (Program) -> Unit = {},
    ) {
        out?.bind()
        GLES20.glUseProgram(p.id)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(target, tex)
        GLES20.glUniform1i(p.u("uTex"), 0)
        GLES20.glUniformMatrix4fv(p.u("uTexMatrix"), 1, false, texMatrix, 0)
        if (out != null) p.v2("uSize", out.width.toFloat(), out.height.toFloat())
        p.f("uTime", time)
        setup(p)
        GLES20.glEnableVertexAttribArray(p.aPos)
        GLES20.glVertexAttribPointer(p.aPos, 2, GLES20.GL_FLOAT, false, 0, pos)
        GLES20.glEnableVertexAttribArray(p.aTex)
        GLES20.glVertexAttribPointer(p.aTex, 2, GLES20.GL_FLOAT, false, 0, uv)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
    }

    fun release() {
        programs.values.forEach { GLES20.glDeleteProgram(it.id) }
        programs.clear()
        a?.release(); b?.release(); a = null; b = null
        stickers.forEach { GLES20.glDeleteTextures(1, intArrayOf(it.second), 0) }
        stickers = emptyList()
        lensName = "none"
        custom.forEach { GLES20.glDeleteTextures(1, intArrayOf(it.second), 0) }
        custom = emptyList()
        customEmoji = emptyList()
        hits = emptyList()
        GLES20.glDeleteTextures(1, intArrayOf(oesTex), 0)
    }

    companion object {
        val IDENTITY = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
        private fun floats(n: Int): FloatBuffer = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        private fun floats(vararg v: Float) = floats(v.size).apply { put(v); position(0) }
        private val QUAD_POS = floats(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        private val QUAD_TEX = floats(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f)
        /** Bitmap rows start at the top; the sprite quad's first two corners are its bottom. */
        private val SPRITE_TEX = floats(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f)
        private val CORNER_X = floatArrayOf(-1f, 1f, -1f, 1f)
        private val CORNER_Y = floatArrayOf(-1f, -1f, 1f, 1f)

        /**
         * A matrix taking an output point (0..1, y up) to the source point: scaled about the center by [sx], [sy]
         * (crop), rotated [degrees] counter-clockwise, then mirrored when [flipX].
         */
        fun texMatrix(sx: Float = 1f, sy: Float = 1f, degrees: Float = 0f, flipX: Boolean = false): FloatArray {
            val m = FloatArray(16)
            Matrix.setIdentityM(m, 0)
            Matrix.translateM(m, 0, 0.5f, 0.5f, 0f)
            if (flipX) Matrix.scaleM(m, 0, -1f, 1f, 1f)
            if (degrees != 0f) Matrix.rotateM(m, 0, degrees, 0f, 0f, 1f)
            Matrix.scaleM(m, 0, sx, sy, 1f)
            Matrix.translateM(m, 0, -0.5f, -0.5f, 0f)
            return m
        }
    }
}
