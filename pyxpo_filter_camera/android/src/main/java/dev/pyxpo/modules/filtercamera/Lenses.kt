package dev.pyxpo.modules.filtercamera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader

internal enum class Anchor { EYES, NOSE, MOUTH, FOREHEAD }

/**
 * A picture stuck to the face. [width] and the offsets [dx] (toward the screen-right eye) and [dy] (toward the
 * forehead) are in face widths.
 */
internal class Sticker(
    val bitmap: Bitmap, val anchor: Anchor, val width: Float, val dx: Float = 0f, val dy: Float = 0f,
    val rotation: Float = 0f,
)

/** One of the app's own stickers (the `stickers` prop): any text, usually an emoji. [rotation] is degrees clockwise. */
internal data class StickerSpec(
    val emoji: String, val anchor: Anchor, val size: Float, val dx: Float, val dy: Float, val rotation: Float,
) {
    companion object {
        fun from(o: org.json.JSONObject) = StickerSpec(
            o.optString("emoji", "?"),
            runCatching { Anchor.valueOf(o.optString("anchor", "eyes").uppercase()) }.getOrDefault(Anchor.EYES),
            o.optDouble("size", 0.4).toFloat(), o.optDouble("dx", 0.0).toFloat(), o.optDouble("dy", 0.0).toFloat(),
            o.optDouble("rotation", 0.0).toFloat(),
        )
    }
}

/** The face lenses, drawn in code (no image files). */
internal object Lenses {
    private const val S = 512f

    fun build(name: String): List<Sticker> = when (name) {
        "puppy" -> listOf(
            Sticker(ear(PUPPY_BROWN, PUPPY_PINK, flop = true, mirror = false), Anchor.FOREHEAD, 0.42f, -0.42f, -0.02f),
            Sticker(ear(PUPPY_BROWN, PUPPY_PINK, flop = true, mirror = true), Anchor.FOREHEAD, 0.42f, 0.42f, -0.02f),
            Sticker(animalNose(Color.rgb(40, 28, 24), 0.62f), Anchor.NOSE, 0.30f, 0f, 0.02f),
            Sticker(tongue(), Anchor.MOUTH, 0.20f, 0f, -0.14f),
        )
        "kitty" -> listOf(
            Sticker(ear(Color.rgb(250, 250, 250), Color.rgb(255, 170, 196), flop = false, mirror = false), Anchor.FOREHEAD, 0.34f, -0.34f, 0.08f),
            Sticker(ear(Color.rgb(250, 250, 250), Color.rgb(255, 170, 196), flop = false, mirror = true), Anchor.FOREHEAD, 0.34f, 0.34f, 0.08f),
            Sticker(whiskers(), Anchor.NOSE, 0.95f, 0f, -0.04f),
            Sticker(animalNose(Color.rgb(255, 140, 170), 0.7f), Anchor.NOSE, 0.13f, 0f, 0.01f),
        )
        "bunny" -> listOf(
            Sticker(longEar(mirror = false), Anchor.FOREHEAD, 0.26f, -0.18f, 0.42f),
            Sticker(longEar(mirror = true), Anchor.FOREHEAD, 0.26f, 0.18f, 0.42f),
            Sticker(animalNose(Color.rgb(255, 150, 175), 0.7f), Anchor.NOSE, 0.12f, 0f, 0.01f),
            Sticker(teeth(), Anchor.MOUTH, 0.16f, 0f, -0.04f),
        )
        "shades" -> listOf(Sticker(shades(), Anchor.EYES, 1.02f, 0f, 0.01f))
        "hearts" -> listOf(Sticker(heartGlasses(), Anchor.EYES, 1.0f, 0f, 0.01f))
        "crown" -> listOf(Sticker(crown(), Anchor.FOREHEAD, 0.72f, 0f, 0.16f))
        "devil" -> listOf(
            Sticker(horn(mirror = false), Anchor.FOREHEAD, 0.22f, -0.27f, 0.12f),
            Sticker(horn(mirror = true), Anchor.FOREHEAD, 0.22f, 0.27f, 0.12f),
        )
        "halo" -> listOf(Sticker(halo(), Anchor.FOREHEAD, 0.8f, 0f, 0.34f))
        "masquerade" -> listOf(Sticker(mask(), Anchor.EYES, 1.08f, 0f, 0.03f))
        "mustache" -> listOf(Sticker(mustache(), Anchor.MOUTH, 0.48f, 0f, 0.09f))
        "clown" -> listOf(
            Sticker(wig(), Anchor.FOREHEAD, 1.55f, 0f, 0.1f),
            Sticker(clownNose(), Anchor.NOSE, 0.2f, 0f, 0.01f),
        )
        else -> emptyList()
    }

    /** [s] drawn with the system (emoji) font, trimmed to the glyphs: one of the app's own stickers. */
    fun emoji(s: String): Bitmap {
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = S * 0.8f; textAlign = Paint.Align.CENTER }
        val m = p.fontMetrics
        val w = (p.measureText(s) + S * 0.1f).coerceIn(S * 0.25f, S * 4)
        val h = m.descent - m.ascent + S * 0.1f
        return canvas(w, h) { drawText(s, w / 2, S * 0.05f - m.ascent, p) }
    }

    private val PUPPY_BROWN = Color.rgb(150, 98, 62)
    private val PUPPY_PINK = Color.rgb(232, 160, 150)

    private fun canvas(w: Float = S, h: Float = S, draw: Canvas.() -> Unit): Bitmap =
        Bitmap.createBitmap(w.toInt(), h.toInt(), Bitmap.Config.ARGB_8888).also { Canvas(it).draw() }

    private fun fill(color: Int) = Paint(Paint.ANTI_ALIAS_FLAG).apply { this.color = color; style = Paint.Style.FILL }

    private fun stroke(color: Int, width: Float) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        this.color = color; style = Paint.Style.STROKE; strokeWidth = width; strokeCap = Paint.Cap.ROUND
    }

    private fun Canvas.mirrored(mirror: Boolean, w: Float, draw: Canvas.() -> Unit) {
        save()
        if (mirror) { translate(w, 0f); scale(-1f, 1f) }
        draw()
        restore()
    }

    private fun ear(outer: Int, inner: Int, flop: Boolean, mirror: Boolean) = canvas(S, S * 1.25f) {
        mirrored(mirror, S) {
            val h = S * 1.25f
            val p = Path()
            if (flop) {
                // a floppy ear hanging from the top of the head
                p.moveTo(S * 0.75f, h * 0.06f)
                p.cubicTo(S * 0.2f, h * 0.0f, S * 0.02f, h * 0.45f, S * 0.12f, h * 0.88f)
                p.cubicTo(S * 0.2f, h * 1.0f, S * 0.45f, h * 0.95f, S * 0.55f, h * 0.7f)
                p.cubicTo(S * 0.65f, h * 0.45f, S * 0.95f, h * 0.3f, S * 0.75f, h * 0.06f)
            } else {
                // a pointed ear
                p.moveTo(S * 0.12f, h * 0.95f)
                p.quadTo(S * 0.2f, h * 0.3f, S * 0.45f, h * 0.05f)
                p.quadTo(S * 0.7f, h * 0.4f, S * 0.92f, h * 0.95f)
                p.close()
            }
            drawPath(p, fill(outer))
            save()
            translate(S * 0.5f, h * 0.55f)
            scale(0.58f, 0.62f)
            translate(-S * 0.5f, -h * 0.55f)
            drawPath(p, fill(inner))
            restore()
        }
    }

    private fun longEar(mirror: Boolean) = canvas(S * 0.5f, S * 1.6f) {
        mirrored(mirror, S * 0.5f) {
            val w = S * 0.5f
            val h = S * 1.6f
            val outer = RectF(w * 0.08f, h * 0.02f, w * 0.92f, h * 0.98f)
            save(); rotate(-6f, w / 2, h); drawOval(outer, fill(Color.rgb(246, 244, 240))); restore()
            val inner = RectF(w * 0.3f, h * 0.12f, w * 0.7f, h * 0.88f)
            save(); rotate(-6f, w / 2, h); drawOval(inner, fill(Color.rgb(255, 182, 200))); restore()
        }
    }

    private fun animalNose(color: Int, height: Float) = canvas(S, S * height) {
        val h = S * height
        val p = Path().apply {
            moveTo(S * 0.5f, h * 0.95f)
            cubicTo(S * 0.15f, h * 0.7f, S * 0.0f, h * 0.15f, S * 0.3f, h * 0.06f)
            quadTo(S * 0.5f, h * 0.0f, S * 0.7f, h * 0.06f)
            cubicTo(S * 1.0f, h * 0.15f, S * 0.85f, h * 0.7f, S * 0.5f, h * 0.95f)
        }
        drawPath(p, fill(color))
        drawOval(RectF(S * 0.28f, h * 0.14f, S * 0.48f, h * 0.32f), fill(Color.argb(110, 255, 255, 255)))
    }

    private fun tongue() = canvas(S * 0.7f, S) {
        val w = S * 0.7f
        val p = Path().apply {
            moveTo(w * 0.1f, 0f)
            lineTo(w * 0.9f, 0f)
            cubicTo(w * 0.95f, S * 0.6f, w * 0.8f, S * 0.95f, w * 0.5f, S * 0.95f)
            cubicTo(w * 0.2f, S * 0.95f, w * 0.05f, S * 0.6f, w * 0.1f, 0f)
        }
        drawPath(p, fill(Color.rgb(240, 96, 120)))
        drawLine(w * 0.5f, S * 0.08f, w * 0.5f, S * 0.6f, stroke(Color.rgb(200, 60, 90), w * 0.05f))
    }

    private fun whiskers() = canvas(S, S * 0.4f) {
        val h = S * 0.4f
        val pen = stroke(Color.argb(230, 40, 30, 30), S * 0.012f)
        for (side in listOf(-1f, 1f)) for (i in 0..2) {
            val y0 = h * (0.38f + i * 0.14f)
            val y1 = h * (0.15f + i * 0.35f)
            val x0 = S / 2 + side * S * 0.12f
            val x1 = S / 2 + side * S * 0.48f
            drawLine(x0, y0, x1, y1, pen)
        }
    }

    private fun teeth() = canvas(S, S * 0.6f) {
        val h = S * 0.6f
        val r = S * 0.06f
        drawRoundRect(RectF(S * 0.05f, 0f, S * 0.48f, h * 0.95f), r, r, fill(Color.WHITE))
        drawRoundRect(RectF(S * 0.52f, 0f, S * 0.95f, h * 0.95f), r, r, fill(Color.WHITE))
        val edge = stroke(Color.rgb(200, 200, 205), S * 0.02f)
        drawRoundRect(RectF(S * 0.05f, 0f, S * 0.48f, h * 0.95f), r, r, edge)
        drawRoundRect(RectF(S * 0.52f, 0f, S * 0.95f, h * 0.95f), r, r, edge)
    }

    private fun shades() = canvas(S, S * 0.36f) {
        val h = S * 0.36f
        val lens = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = LinearGradient(0f, 0f, 0f, h, Color.argb(240, 30, 34, 48), Color.argb(225, 70, 40, 90), Shader.TileMode.CLAMP)
        }
        val left = RectF(S * 0.04f, h * 0.12f, S * 0.45f, h * 0.92f)
        val right = RectF(S * 0.55f, h * 0.12f, S * 0.96f, h * 0.92f)
        drawRoundRect(left, h * 0.3f, h * 0.3f, lens)
        drawRoundRect(right, h * 0.3f, h * 0.3f, lens)
        val frame = stroke(Color.rgb(15, 15, 18), S * 0.025f)
        drawRoundRect(left, h * 0.3f, h * 0.3f, frame)
        drawRoundRect(right, h * 0.3f, h * 0.3f, frame)
        drawLine(S * 0.45f, h * 0.35f, S * 0.55f, h * 0.35f, frame)
        val shine = stroke(Color.argb(120, 255, 255, 255), S * 0.018f)
        drawLine(S * 0.1f, h * 0.35f, S * 0.18f, h * 0.25f, shine)
        drawLine(S * 0.61f, h * 0.35f, S * 0.69f, h * 0.25f, shine)
    }

    private fun heart(cx: Float, cy: Float, r: Float) = Path().apply {
        moveTo(cx, cy + r * 0.95f)
        cubicTo(cx - r * 1.6f, cy - r * 0.1f, cx - r * 0.9f, cy - r * 1.35f, cx, cy - r * 0.5f)
        cubicTo(cx + r * 0.9f, cy - r * 1.35f, cx + r * 1.6f, cy - r * 0.1f, cx, cy + r * 0.95f)
    }

    private fun heartGlasses() = canvas(S, S * 0.42f) {
        val h = S * 0.42f
        val glass = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = LinearGradient(0f, 0f, 0f, h, Color.argb(225, 255, 70, 130), Color.argb(225, 200, 20, 80), Shader.TileMode.CLAMP)
        }
        val frame = stroke(Color.rgb(255, 230, 240), S * 0.018f)
        for (cx in listOf(S * 0.25f, S * 0.75f)) {
            val p = heart(cx, h * 0.52f, h * 0.48f)
            drawPath(p, glass)
            drawPath(p, frame)
        }
        drawLine(S * 0.44f, h * 0.38f, S * 0.56f, h * 0.38f, frame)
    }

    private fun crown() = canvas(S, S * 0.6f) {
        val h = S * 0.6f
        val gold = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = LinearGradient(0f, 0f, 0f, h, Color.rgb(255, 226, 110), Color.rgb(214, 150, 30), Shader.TileMode.CLAMP)
        }
        val p = Path().apply {
            moveTo(S * 0.06f, h * 0.95f)
            lineTo(S * 0.02f, h * 0.25f)
            lineTo(S * 0.27f, h * 0.55f)
            lineTo(S * 0.5f, h * 0.05f)
            lineTo(S * 0.73f, h * 0.55f)
            lineTo(S * 0.98f, h * 0.25f)
            lineTo(S * 0.94f, h * 0.95f)
            close()
        }
        drawPath(p, gold)
        drawPath(p, stroke(Color.rgb(170, 110, 20), S * 0.012f))
        val gems = listOf(Color.rgb(230, 40, 70), Color.rgb(40, 140, 240), Color.rgb(40, 200, 120))
        drawCircle(S * 0.5f, h * 0.72f, S * 0.05f, fill(gems[0]))
        drawCircle(S * 0.27f, h * 0.76f, S * 0.035f, fill(gems[1]))
        drawCircle(S * 0.73f, h * 0.76f, S * 0.035f, fill(gems[2]))
        for (x in listOf(0.02f, 0.5f, 0.98f)) drawCircle(S * x, h * (if (x == 0.5f) 0.05f else 0.25f), S * 0.03f, fill(Color.rgb(255, 245, 200)))
    }

    private fun horn(mirror: Boolean) = canvas(S * 0.6f, S) {
        mirrored(mirror, S * 0.6f) {
            val w = S * 0.6f
            val p = Path().apply {
                moveTo(w * 0.15f, S * 0.98f)
                cubicTo(w * 0.05f, S * 0.6f, w * 0.15f, S * 0.25f, w * 0.75f, S * 0.02f)
                cubicTo(w * 0.45f, S * 0.35f, w * 0.75f, S * 0.7f, w * 0.9f, S * 0.98f)
                close()
            }
            val red = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                shader = LinearGradient(0f, S, w, 0f, Color.rgb(150, 0, 20), Color.rgb(255, 60, 60), Shader.TileMode.CLAMP)
            }
            drawPath(p, red)
        }
    }

    private fun halo() = canvas(S, S * 0.32f) {
        val h = S * 0.32f
        val oval = RectF(S * 0.06f, h * 0.15f, S * 0.94f, h * 0.85f)
        drawOval(oval, stroke(Color.argb(90, 255, 240, 160), S * 0.07f))
        drawOval(oval, stroke(Color.rgb(255, 222, 90), S * 0.035f))
        drawOval(oval, stroke(Color.argb(200, 255, 255, 230), S * 0.01f))
    }

    private fun mask() = canvas(S, S * 0.44f) {
        val h = S * 0.44f
        val p = Path().apply {
            moveTo(S * 0.5f, h * 0.3f)
            cubicTo(S * 0.35f, h * 0.0f, S * 0.05f, h * 0.0f, S * 0.0f, h * 0.2f)
            cubicTo(S * 0.02f, h * 0.7f, S * 0.25f, h * 0.98f, S * 0.42f, h * 0.8f)
            quadTo(S * 0.5f, h * 0.7f, S * 0.58f, h * 0.8f)
            cubicTo(S * 0.75f, h * 0.98f, S * 0.98f, h * 0.7f, S * 1.0f, h * 0.2f)
            cubicTo(S * 0.95f, h * 0.0f, S * 0.65f, h * 0.0f, S * 0.5f, h * 0.3f)
        }
        val purple = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = LinearGradient(0f, 0f, S, h, Color.rgb(110, 40, 170), Color.rgb(40, 20, 90), Shader.TileMode.CLAMP)
        }
        drawPath(p, purple)
        drawPath(p, stroke(Color.rgb(240, 200, 90), S * 0.014f))
        val hole = Paint(Paint.ANTI_ALIAS_FLAG).apply { xfermode = android.graphics.PorterDuffXfermode(android.graphics.PorterDuff.Mode.CLEAR) }
        drawOval(RectF(S * 0.12f, h * 0.32f, S * 0.4f, h * 0.68f), hole)
        drawOval(RectF(S * 0.6f, h * 0.32f, S * 0.88f, h * 0.68f), hole)
        val rim = stroke(Color.rgb(240, 200, 90), S * 0.01f)
        drawOval(RectF(S * 0.12f, h * 0.32f, S * 0.4f, h * 0.68f), rim)
        drawOval(RectF(S * 0.6f, h * 0.32f, S * 0.88f, h * 0.68f), rim)
    }

    private fun mustache() = canvas(S, S * 0.34f) {
        val h = S * 0.34f
        val p = Path().apply {
            moveTo(S * 0.5f, h * 0.25f)
            cubicTo(S * 0.38f, h * 0.0f, S * 0.18f, h * 0.1f, S * 0.12f, h * 0.45f)
            cubicTo(S * 0.06f, h * 0.65f, S * 0.0f, h * 0.55f, S * 0.02f, h * 0.35f)
            cubicTo(S * 0.0f, h * 0.9f, S * 0.32f, h * 1.0f, S * 0.5f, h * 0.65f)
            cubicTo(S * 0.68f, h * 1.0f, S * 1.0f, h * 0.9f, S * 0.98f, h * 0.35f)
            cubicTo(S * 1.0f, h * 0.55f, S * 0.94f, h * 0.65f, S * 0.88f, h * 0.45f)
            cubicTo(S * 0.82f, h * 0.1f, S * 0.62f, h * 0.0f, S * 0.5f, h * 0.25f)
        }
        drawPath(p, fill(Color.rgb(48, 32, 24)))
    }

    private fun clownNose() = canvas {
        val ball = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = RadialGradient(S * 0.38f, S * 0.36f, S * 0.6f, Color.rgb(255, 90, 90), Color.rgb(180, 0, 10), Shader.TileMode.CLAMP)
        }
        drawCircle(S / 2, S / 2, S * 0.46f, ball)
        drawOval(RectF(S * 0.26f, S * 0.2f, S * 0.44f, S * 0.34f), fill(Color.argb(150, 255, 255, 255)))
    }

    private fun wig() = canvas(S, S * 0.55f) {
        val h = S * 0.55f
        val colors = intArrayOf(
            Color.rgb(255, 70, 70), Color.rgb(255, 170, 40), Color.rgb(255, 230, 60),
            Color.rgb(70, 210, 110), Color.rgb(60, 150, 255), Color.rgb(170, 90, 240),
        )
        // puffs around the top of the head; the middle stays clear for the face
        val puffs = listOf(
            0.1f to 0.75f, 0.06f to 0.5f, 0.14f to 0.28f, 0.27f to 0.14f, 0.42f to 0.08f,
            0.58f to 0.08f, 0.73f to 0.14f, 0.86f to 0.28f, 0.94f to 0.5f, 0.9f to 0.75f,
        )
        puffs.forEachIndexed { i, (x, y) -> drawCircle(S * x, h * y, S * 0.1f, fill(colors[i % colors.size])) }
        puffs.forEachIndexed { i, (x, y) -> drawCircle(S * x, h * y + S * 0.04f, S * 0.075f, fill(colors[(i + 2) % colors.size])) }
    }
}
