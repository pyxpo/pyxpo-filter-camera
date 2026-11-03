package dev.pyxpo.modules.filtercamera

import org.json.JSONObject

/**
 * GLSL for the filter passes. Every fragment shader exists twice: reading the camera's external (OES) texture and
 * reading a normal 2D texture (an earlier pass). `SRC(p)` hides the difference: it samples the source at quad
 * coordinate `p` (0..1, y up) through `uTexMatrix`.
 */
object Shaders {
    const val VERTEX = """
attribute vec4 aPos;
attribute vec2 aTex;
varying vec2 vUv;
void main() { gl_Position = aPos; vUv = aTex; }
"""

    private const val OES_HEAD = "#extension GL_OES_EGL_image_external : require\nprecision highp float;\nuniform samplerExternalOES uTex;\n"
    private const val TEX_HEAD = "precision highp float;\nuniform sampler2D uTex;\n"
    private const val COMMON = """
uniform mat4 uTexMatrix;
uniform vec2 uSize;
uniform float uTime;
varying vec2 vUv;
vec4 SRC(vec2 p) { return texture2D(uTex, (uTexMatrix * vec4(p, 0.0, 1.0)).xy); }
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
"""

    fun fragment(body: String, oes: Boolean) = (if (oes) OES_HEAD else TEX_HEAD) + COMMON + body

    const val COPY = "void main() { gl_FragColor = SRC(vUv); }"

    /** Face warps (slim, bigger eyes), then edge-keeping skin smoothing, then brightening. Face points are pixels, y up. */
    const val BEAUTY = """
uniform float uSmooth;
uniform float uBright;
uniform float uSlim;
uniform float uEyes;
uniform float uFace;
uniform vec2 uEyeA;
uniform vec2 uEyeB;
uniform vec2 uCenter;
uniform vec2 uAxX;
uniform vec2 uAxY;
uniform vec2 uRad;
uniform float uEyeRad;
uniform float uRadius;

vec2 magnify(vec2 px, vec2 c) {
    vec2 d = px - c;
    float r = length(d) / uEyeRad;
    if (r < 1.0) px = c + d * (1.0 - uEyes * 0.22 * (1.0 - r * r));
    return px;
}

vec2 warp(vec2 px) {
    if (uFace < 0.5) return px;
    if (uEyes > 0.0) { px = magnify(px, uEyeA); px = magnify(px, uEyeB); }
    if (uSlim > 0.0) {
        vec2 q = vec2(dot(px - uCenter, uAxX), dot(px - uCenter, uAxY));
        vec2 e = q / uRad;
        float r2 = dot(e, e);
        if (r2 < 1.0) {
            float w = (1.0 - r2) * (1.0 - r2);
            float lower = q.y < 0.0 ? 1.0 : max(0.0, 1.0 - 2.0 * q.y / uRad.y);
            q.x *= 1.0 + uSlim * 0.2 * w * lower;
            px = uCenter + q.x * uAxX + q.y * uAxY;
        }
    }
    return px;
}

void main() {
    vec2 uv = warp(vUv * uSize) / uSize;
    vec3 c = SRC(uv).rgb;
    if (uSmooth > 0.0) {
        vec2 st = vec2(uRadius) / uSize;
        vec3 sum = c;
        float ws = 1.0;
        for (int i = 0; i < 8; i++) {
            float a = float(i) * 0.7853982;
            vec2 o = vec2(cos(a), sin(a)) * st;
            vec3 s1 = SRC(uv + o).rgb;
            vec3 d1 = s1 - c;
            float w1 = max(0.0, 1.0 - dot(d1, d1) * 14.0);
            vec3 s2 = SRC(uv + vec2(cos(a + 0.3926991), sin(a + 0.3926991)) * st * 2.0).rgb;
            vec3 d2 = s2 - c;
            float w2 = max(0.0, 1.0 - dot(d2, d2) * 14.0) * 0.7;
            sum += s1 * w1 + s2 * w2;
            ws += w1 + w2;
        }
        vec3 sm = sum / ws;
        float cb = 0.5 - 0.168736 * c.r - 0.331264 * c.g + 0.5 * c.b;
        float cr = 0.5 + 0.5 * c.r - 0.418688 * c.g - 0.081312 * c.b;
        float skin = smoothstep(0.27, 0.31, cb) * (1.0 - smoothstep(0.49, 0.53, cb))
                   * smoothstep(0.50, 0.53, cr) * (1.0 - smoothstep(0.67, 0.71, cr));
        c = mix(c, sm, uSmooth * mix(0.25, 0.95, skin));
    }
    if (uBright > 0.0) c = pow(c, vec3(1.0 - uBright * 0.3)) + uBright * 0.02;
    gl_FragColor = vec4(c, 1.0);
}
"""

    /** One color grade with parameters; every filter preset is a set of these. */
    const val COLOR = """
uniform float uIntensity;
uniform float uExposure;
uniform float uContrast;
uniform float uSaturation;
uniform float uTemp;
uniform float uTint;
uniform float uFade;
uniform vec3 uShadow;
uniform vec3 uHigh;
uniform float uSplit;
uniform float uVignette;
uniform float uGrain;
uniform float uMono;
uniform float uSepia;
uniform float uHue;
uniform vec4 uTones;      // highlights, shadows, whites, blacks
uniform vec4 uPresence;   // vibrance, clarity, sharpen, dehaze
uniform vec3 uHsl[8];     // hue, saturation, luminance of red, orange, yellow, green, aqua, blue, purple, magenta
uniform float uHslOn;

vec3 hueRotate(vec3 c, float a) {
    float k = cos(a), q = sin(a);
    return vec3(
        dot(c, vec3(0.213 + k * 0.787 - q * 0.213, 0.715 - k * 0.715 - q * 0.715, 0.072 - k * 0.072 + q * 0.928)),
        dot(c, vec3(0.213 - k * 0.213 + q * 0.143, 0.715 + k * 0.285 + q * 0.140, 0.072 - k * 0.072 - q * 0.283)),
        dot(c, vec3(0.213 - k * 0.213 - q * 0.787, 0.715 - k * 0.715 + q * 0.715, 0.072 + k * 0.928 + q * 0.072)));
}

// four diagonal samples of the source, r pixels out: a cheap blur for clarity and sharpen
vec3 around(float r) {
    vec2 o = r / uSize;
    return (SRC(vUv + o).rgb + SRC(vUv - o).rgb + SRC(vUv + vec2(o.x, -o.y)).rgb + SRC(vUv + vec2(-o.x, o.y)).rgb) * 0.25;
}

// the eight ranges sit at 0, 30, 60, 120, 180, 240, 270, 300 degrees; h is in units of 60 degrees
vec3 hslAt(float h) {
    if (h < 0.5) return mix(uHsl[0], uHsl[1], h / 0.5);
    if (h < 1.0) return mix(uHsl[1], uHsl[2], (h - 0.5) / 0.5);
    if (h < 2.0) return mix(uHsl[2], uHsl[3], h - 1.0);
    if (h < 3.0) return mix(uHsl[3], uHsl[4], h - 2.0);
    if (h < 4.0) return mix(uHsl[4], uHsl[5], h - 3.0);
    if (h < 4.5) return mix(uHsl[5], uHsl[6], (h - 4.0) / 0.5);
    if (h < 5.0) return mix(uHsl[6], uHsl[7], (h - 4.5) / 0.5);
    return mix(uHsl[7], uHsl[0], h - 5.0);
}

vec3 selective(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    float hi = max(c.r, max(c.g, c.b)), d = hi - min(c.r, min(c.g, c.b));
    if (d < 0.001) return c;
    float h = hi == c.r ? mod((c.g - c.b) / d, 6.0) : hi == c.g ? (c.b - c.r) / d + 2.0 : (c.r - c.g) / d + 4.0;
    float s = d / hi;
    vec3 a = hslAt(h);
    float k = smoothstep(0.0, 0.2, s);            // near-grey pixels barely move
    h = mod(h + a.x * k * 0.5, 6.0);              // up to 30 degrees either way
    s = clamp(s * (1.0 + a.y * k), 0.0, 1.0);
    vec3 p = clamp(abs(mod(h + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    vec3 rgb = hi * mix(vec3(1.0), p, s);
    rgb *= 1.0 + min(a.z, 0.0) * k * 0.6;
    return rgb + (1.0 - rgb) * max(a.z, 0.0) * k * 0.5;
}

void main() {
    vec3 src = SRC(vUv).rgb;
    vec3 c = src;
    // detail goes through luma only, so hard colour edges and skin get no colour fringe
    if (uPresence.z != 0.0) c += luma(c - around(uSize.y / 850.0)) * uPresence.z * 1.5;
    if (uPresence.y != 0.0) {
        float m = luma(clamp(c, 0.0, 1.0));
        c += clamp(luma(c - around(uSize.y / 180.0)), -0.2, 0.2) * uPresence.y * 4.0 * m * (1.0 - m);
    }
    c *= exp2(uExposure);
    float t = luma(clamp(c, 0.0, 1.0));
    c += uTones.x * 0.25 * smoothstep(0.4, 0.75, t) * (1.0 - smoothstep(0.75, 1.0, t))
       + uTones.y * 0.25 * smoothstep(0.0, 0.25, t) * (1.0 - smoothstep(0.25, 0.6, t))
       + uTones.z * 0.18 * smoothstep(0.75, 1.0, t)
       + uTones.w * 0.18 * (1.0 - smoothstep(0.0, 0.25, t));
    if (uPresence.w > 0.0) {
        float k = uPresence.w * 0.12;
        c = (c - k) / (1.0 - k);
        c = mix(vec3(luma(c)), c, 1.0 + uPresence.w * 0.2);
    } else if (uPresence.w < 0.0) {
        c = mix(c, vec3(0.75) + c * 0.25, -uPresence.w * 0.35);
    }
    c = (c - 0.5) * uContrast + 0.5;
    float l = luma(c);
    c = mix(vec3(l), c, uSaturation);
    if (uPresence.x != 0.0) {
        float sat = clamp(max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b)), 0.0, 1.0);
        c = mix(vec3(l), c, 1.0 + uPresence.x * (1.0 - sat));
    }
    if (uHslOn > 0.5) c = selective(c);
    if (uSepia > 0.0) c = mix(c, vec3(dot(c, vec3(0.393, 0.769, 0.189)), dot(c, vec3(0.349, 0.686, 0.168)),
                                      dot(c, vec3(0.272, 0.534, 0.131))), uSepia);
    if (uHue != 0.0) c = hueRotate(c, uHue);
    c += vec3(uTemp * 0.08, uTint * 0.06, -uTemp * 0.08);
    l = luma(c);
    c += mix(uShadow, uHigh, smoothstep(0.0, 1.0, l)) * uSplit;
    c = mix(c, vec3(0.07) + c * 0.88, uFade);
    c = mix(c, vec3(luma(c)), uMono);
    vec2 d = (vUv - 0.5) * vec2(uSize.x / uSize.y, 1.0);
    c *= 1.0 - uVignette * smoothstep(0.35, 0.95, length(d) * 1.25);
    c += (hash(floor(vUv * uSize) + fract(uTime) * 91.0) - 0.5) * uGrain * 0.14;
    gl_FragColor = vec4(mix(src, clamp(c, 0.0, 1.0), uIntensity), 1.0);
}
"""

    private const val EDGE = """
float L(vec2 p) { return luma(SRC(p).rgb); }
float edge(vec2 uv) {
    vec2 o = vec2(uSize.y / 1280.0) / uSize;
    float tl = L(uv + vec2(-o.x, o.y)), t = L(uv + vec2(0.0, o.y)), tr = L(uv + o);
    float l = L(uv - vec2(o.x, 0.0)), r = L(uv + vec2(o.x, 0.0));
    float bl = L(uv - o), b = L(uv - vec2(0.0, o.y)), br = L(uv + vec2(o.x, -o.y));
    float gx = tr + 2.0 * r + br - tl - 2.0 * l - bl;
    float gy = tl + 2.0 * t + tr - bl - 2.0 * b - br;
    return length(vec2(gx, gy));
}
"""

    private const val FX_MAIN = """
uniform float uAmount;
void main() { gl_FragColor = vec4(mix(SRC(vUv).rgb, fx(vUv), uAmount), 1.0); }
"""

    private val EFFECTS = mapOf(
        "glitch" to """
vec3 fx(vec2 uv) {
    float t = floor(uTime * 12.0);
    float r = hash(vec2(floor(uv.y * 24.0), t));
    float burst = step(0.82, hash(vec2(t, 7.0)));
    float shift = (r - 0.5) * 0.14 * step(0.62, r) * (0.25 + burst);
    vec2 p = vec2(fract(uv.x + shift), uv.y);
    float s = 0.005 + 0.02 * burst;
    return vec3(SRC(p + vec2(s, 0.0)).r, SRC(p).g, SRC(p - vec2(s, 0.0)).b);
}""",
        "rgb_split" to """
vec3 fx(vec2 uv) {
    vec2 o = (uv - 0.5) * 0.035 + vec2(0.006, 0.0);
    return vec3(SRC(uv + o).r, SRC(uv).g, SRC(uv - o).b);
}""",
        "mirror" to """
vec3 fx(vec2 uv) { return SRC(vec2(uv.x > 0.5 ? 1.0 - uv.x : uv.x, uv.y)).rgb; }""",
        "vhs" to """
vec3 fx(vec2 uv) {
    float wob = sin(uv.y * 30.0 + uTime * 3.0) * 0.002 + (hash(vec2(floor(uv.y * 240.0), floor(uTime * 30.0))) - 0.5) * 0.004;
    vec2 p = vec2(uv.x + wob, uv.y);
    vec3 c = vec3(SRC(p + vec2(0.004, 0.0)).r, SRC(p).g, SRC(p - vec2(0.004, 0.0)).b);
    c *= 0.88 + 0.12 * sin(uv.y * uSize.y * 1.4);
    c += (hash(uv * uSize + fract(uTime) * 57.0) - 0.5) * 0.09;
    return mix(vec3(luma(c)), c, 0.75) * vec3(1.06, 1.0, 0.9);
}""",
        "pixelate" to """
vec3 fx(vec2 uv) {
    float b = max(4.0, uSize.x / 48.0);
    return SRC((floor(uv * uSize / b) + 0.5) * b / uSize).rgb;
}""",
        "kaleido" to """
vec3 fx(vec2 uv) {
    float asp = uSize.x / uSize.y;
    vec2 d = uv - 0.5; d.x *= asp;
    float a = atan(d.y, d.x) + uTime * 0.15;
    float seg = 6.2831853 / 6.0;
    a = mod(a, seg); a = abs(a - seg * 0.5);
    d = vec2(cos(a), sin(a)) * length(d); d.x /= asp;
    return SRC(d + 0.5).rgb;
}""",
        "thermal" to """
vec3 fx(vec2 uv) {
    float l = luma(SRC(uv).rgb);
    vec3 a = vec3(0.0, 0.0, 0.35), b = vec3(0.55, 0.0, 0.75), c = vec3(1.0, 0.15, 0.0), d = vec3(1.0, 0.9, 0.0);
    if (l < 0.3) return mix(a, b, l / 0.3);
    if (l < 0.55) return mix(b, c, (l - 0.3) / 0.25);
    if (l < 0.8) return mix(c, d, (l - 0.55) / 0.25);
    return mix(d, vec3(1.0), (l - 0.8) / 0.2);
}""",
        "comic" to EDGE + """
vec3 fx(vec2 uv) {
    vec3 c = SRC(uv).rgb;
    c = mix(vec3(luma(c)), c, 1.5);
    c = floor(clamp(c, 0.0, 1.0) * 5.0 + 0.5) / 5.0;
    return c * (1.0 - smoothstep(0.25, 0.45, edge(uv)));
}""",
        "sketch" to EDGE + """
vec3 fx(vec2 uv) {
    float g = 1.0 - smoothstep(0.06, 0.4, edge(uv));
    float shade = mix(0.75, 1.0, smoothstep(0.1, 0.6, luma(SRC(uv).rgb)));
    return vec3(0.97, 0.96, 0.92) * g * shade;
}""",
        "neon" to EDGE + """
vec3 fx(vec2 uv) {
    float e = smoothstep(0.08, 0.5, edge(uv));
    vec3 hue = 0.5 + 0.5 * cos(6.2831853 * (uv.y * 0.6 + uTime * 0.2 + vec3(0.0, 0.33, 0.67)));
    return SRC(uv).rgb * 0.12 + hue * e * 1.6;
}""",
        "posterize" to """
vec3 fx(vec2 uv) {
    vec3 c = SRC(uv).rgb;
    c = mix(vec3(luma(c)), c, 1.3);
    return floor(clamp(c, 0.0, 1.0) * 4.0 + 0.5) / 4.0;
}""",
        "bulge" to """
vec3 fx(vec2 uv) {
    float asp = uSize.x / uSize.y;
    vec2 d = uv - 0.5; d.x *= asp;
    float r = length(d);
    if (r < 0.42) d *= mix(0.55, 1.0, (r / 0.42) * (r / 0.42));
    d.x /= asp;
    return SRC(d + 0.5).rgb;
}""",
        "wave" to """
vec3 fx(vec2 uv) {
    vec2 p = uv + vec2(sin(uv.y * 18.0 + uTime * 3.0), cos(uv.x * 14.0 + uTime * 2.5)) * 0.012;
    return SRC(p).rgb;
}""",
    )

    fun effect(name: String): String? = EFFECTS[name]?.let { it + FX_MAIN }
}

/** A color grade: the uniforms of [Shaders.COLOR]. */
class Grade(
    val exposure: Float = 0f, val contrast: Float = 1f, val saturation: Float = 1f, val temperature: Float = 0f,
    val tint: Float = 0f, val fade: Float = 0f, val shadow: FloatArray = ZERO, val highlight: FloatArray = ZERO,
    val split: Float = 0f, val vignette: Float = 0f, val grain: Float = 0f, val mono: Float = 0f,
    val sepia: Float = 0f, val hue: Float = 0f, val vibrance: Float = 0f, val toneHighlights: Float = 0f,
    val toneShadows: Float = 0f, val whites: Float = 0f, val blacks: Float = 0f, val clarity: Float = 0f,
    val dehaze: Float = 0f, val sharpen: Float = 0f, val hsl: FloatArray = NO_HSL,
) {
    /** Skips the per-pixel colour-range work when no range was touched. */
    val hslOn: Boolean = hsl.any { it != 0f }

    companion object {
        private val ZERO = floatArrayOf(0f, 0f, 0f)
        private val NO_HSL = FloatArray(24)

        /** The `grade` prop: Python fills every key and sends `hue` in degrees. */
        fun from(o: JSONObject): Grade {
            fun f(k: String, d: Float) = o.optDouble(k, d.toDouble()).toFloat()
            fun c(k: String) = o.optJSONArray(k)?.takeIf { it.length() == 3 }
                ?.let { a -> FloatArray(3) { a.optDouble(it, 0.0).toFloat() } } ?: ZERO
            val hsl = o.optJSONArray("hsl")?.takeIf { it.length() == 24 }
                ?.let { a -> FloatArray(24) { a.optDouble(it, 0.0).toFloat() } } ?: NO_HSL
            return Grade(
                exposure = f("exposure", 0f), contrast = f("contrast", 1f), saturation = f("saturation", 1f),
                temperature = f("temperature", 0f), tint = f("tint", 0f), fade = f("fade", 0f),
                shadow = c("shadow"), highlight = c("highlight"), split = f("split", 0f),
                vignette = f("vignette", 0f), grain = f("grain", 0f), mono = f("mono", 0f),
                sepia = f("sepia", 0f), hue = f("hue", 0f) * 0.017453292f, vibrance = f("vibrance", 0f),
                toneHighlights = f("tone_highlights", 0f), toneShadows = f("tone_shadows", 0f),
                whites = f("whites", 0f), blacks = f("blacks", 0f), clarity = f("clarity", 0f),
                dehaze = f("dehaze", 0f), sharpen = f("sharpen", 0f), hsl = hsl,
            )
        }
    }
}
