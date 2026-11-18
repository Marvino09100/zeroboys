"""Modern future-bass style radio jingle, synthesized from scratch (numpy + scipy).

Outputs (WAV, 44.1 kHz stereo):
  zeroboys_jingle.wav        full ident  (~7 s)  build-up -> drop -> tape stop
  zeroboys_jingle_short.wav  stinger     (~3 s)  for between songs
  zeroboys_bed.wav           music bed   (~13 s) to talk over (no lead melody)
"""
import wave
import numpy as np
from scipy.signal import lfilter, butter, sosfilt

SR = 44100
BPM = 150
BEAT = 60 / BPM
rng = np.random.default_rng(42)


# ---------- helpers ----------
def t(dur):
    return np.arange(int(dur * SR)) / SR


def hz(n):
    return 440 * 2 ** ((n - 69) / 12)


def adsr(n, a=0.005, d=0.1, s=0.7, r=0.05):
    x = np.full(n, s, dtype=float)
    na = max(1, min(int(a * SR), n)); x[:na] = np.linspace(0, 1, na)
    nd = min(int(d * SR), n - na); x[na:na + nd] = np.linspace(1, s, nd)
    nr = min(int(r * SR), n); x[n - nr:] *= np.linspace(1, 0, nr)
    return x


def place(buf, sig, at, gain=1.0):
    i = int(round(at * SR))
    if i >= len(buf): return
    end = min(len(buf), i + len(sig))
    buf[i:end] += sig[:end - i] * gain


def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR / 2 - 100) / (SR / 2), "low", output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc / (SR / 2), "high", output="sos"), x)


def sweep_lp(x, f0, f1):
    """Low-pass sweep, processed in blocks."""
    out = np.zeros_like(x); blk = 1024; zi = None
    nb = len(x) // blk + 1
    for k in range(nb):
        fc = f0 * (f1 / f0) ** (k / max(1, nb - 1))
        sos = butter(2, min(fc, SR / 2 - 100) / (SR / 2), "low", output="sos")
        if zi is None: zi = np.zeros((sos.shape[0], 2))
        seg = x[k * blk:(k + 1) * blk]
        if len(seg) == 0: break
        out[k * blk:k * blk + len(seg)], zi = sosfilt(sos, seg, zi=zi)
    return out


def bandpass(x, fc, q):
    w0 = 2 * np.pi * fc / SR; alpha = np.sin(w0) / (2 * q)
    b = [alpha, 0, -alpha]; a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
    return lfilter(b, a, x)


def saw(freq_arr):
    ph = np.cumsum(freq_arr) / SR + rng.random()
    return 2 * (ph % 1) - 1


def supersaw(f, dur, voices=7, detune=0.018):
    n = int(dur * SR); out = np.zeros(n)
    for i in range(voices):
        d = 1 + detune * (i - voices // 2) / (voices // 2)
        out += saw(np.full(n, f * d))
    return out / np.sqrt(voices)


# ---------- instruments ----------
VOWELS = {"a": [(800, 1.0), (1150, 0.5), (2900, 0.25)],
          "o": [(450, 1.0), (800, 0.6), (2830, 0.15)],
          "e": [(300, 1.0), (2250, 0.6), (3000, 0.35)],
          "i": [(270, 1.0), (2300, 0.8), (3100, 0.4)]}


def chop(note, dur, vowel="a", bend=True):
    """Pitched 'vocal chop' — formant-filtered saw with a short pitch scoop."""
    n = int(dur * SR); tt = np.arange(n) / SR
    f = hz(note) * np.ones(n)
    if bend:
        f *= 2 ** ((-1.0 * np.exp(-tt * 40)) / 12)  # scoop up a semitone
    f *= 1 + 0.004 * np.sin(2 * np.pi * 5.5 * tt)
    src = saw(f) + 0.08 * rng.standard_normal(n)
    out = sum(g * bandpass(src, fc, 6) for fc, g in VOWELS[vowel])
    out = out * adsr(n, a=0.004, d=0.06, s=0.8, r=0.03)
    return out / (np.max(np.abs(out)) + 1e-9)


def pluck(note, dur):
    n = int(dur * SR); tt = np.arange(n) / SR
    s = supersaw(hz(note), dur, voices=3, detune=0.008)
    out = np.zeros(n); fc = 300 + 5000 * np.exp(-tt * 18)
    blk = 256
    for k in range(0, n, blk):
        out[k:k + blk] = lp(s[k:k + blk], fc[k], 1)
    return out * np.exp(-tt * 6)


def kick(dur=0.35):
    tt = t(dur)
    f = 50 + 180 * np.exp(-tt * 35)
    k = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 9)
    click = hp(rng.standard_normal(len(tt)), 3000) * np.exp(-tt * 300) * 0.3
    return np.tanh(2 * (k + click))


def bass808(note, dur):
    tt = t(dur)
    f = hz(note) * (1 + 1.5 * np.exp(-tt * 30))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR)
    s = np.tanh(2.5 * s) * adsr(len(tt), a=0.002, d=0.3, s=0.75, r=0.08)
    return s


def clap(dur=0.35):
    tt = t(dur); n = len(tt)
    noise = bandpass(rng.standard_normal(n), 1500, 1.2)
    e = np.zeros(n)
    for off in (0, 0.011, 0.022):
        i = int(off * SR); e[i:] = np.maximum(e[i:], np.exp(-(tt[:n - i]) * (70 if off < 0.02 else 14)))
    return noise * e * 1.6


def hat(dur=0.05, open_=False):
    tt = t(0.25 if open_ else dur)
    n = hp(rng.standard_normal(len(tt)), 8000, 4)
    return n * np.exp(-tt * (12 if open_ else 90)) * 0.5


def noise_riser(dur, up=True):
    n = rng.standard_normal(int(dur * SR))
    out = sweep_lp(n, 200, 12000) if up else sweep_lp(n, 12000, 200)
    ramp = np.linspace(0, 1, len(n)) ** 2
    return out * (ramp if up else ramp[::-1])


def reverse_cymbal(dur):
    tt = t(dur)
    n = hp(rng.standard_normal(len(tt)), 5000) * np.exp(-tt * 3)
    return n[::-1] * 0.5 * np.linspace(0, 1, len(n)) ** 2


def impact(dur=2.0):
    tt = t(dur)
    boom = np.sin(2 * np.pi * np.cumsum(40 + 60 * np.exp(-tt * 10)) / SR) * np.exp(-tt * 2.5)
    shh = hp(rng.standard_normal(len(tt)), 4000) * np.exp(-tt * 2)
    return boom * 0.8 + shh * 0.25


# ---------- mixing utilities ----------
def sidechain(length, kick_times, depth=0.85, release=0.22):
    g = np.ones(length)
    rel = np.exp(-np.arange(int(release * 3 * SR)) / (release * SR / 3))
    for kt in kick_times:
        i = int(kt * SR); seg = 1 - depth * rel
        end = min(length, i + len(seg))
        g[i:end] = np.minimum(g[i:end], seg[:end - i])
    return g


def reverb(x, mix=0.25, size=1.0):
    out = np.zeros((len(x), 2))
    combs = [(0.0297, 0.0371, 0.0411, 0.0437), (0.0311, 0.0353, 0.0429, 0.0449)]
    for ch in range(2):
        wet = np.zeros_like(x)
        for d in combs[ch]:
            n = int(d * size * SR); g = 0.78
            wet += lfilter([1], np.concatenate([[1], np.zeros(n - 1), [-g]]), x)
        for d in (0.005, 0.0017):  # allpass diffusion
            n = int(d * SR); g = 0.6
            b = np.concatenate([[-g], np.zeros(n - 1), [1]]); a = np.concatenate([[1], np.zeros(n - 1), [-g]])
            wet = lfilter(b, a, wet)
        wet = hp(lp(wet, 7000), 250)
        out[:, ch] = x + mix * wet / 4
    return out


def tape_stop(x, start, dur):
    """Slow playback down to zero from `start` seconds over `dur` seconds (stereo)."""
    i0 = int(start * SR); n = int(dur * SR)
    rate = np.linspace(1, 0, n) ** 1.5
    pos = i0 + np.cumsum(rate)
    out = x[:i0 + n].copy()
    for ch in range(x.shape[1]):
        out[i0:i0 + n, ch] = np.interp(pos, np.arange(len(x)), x[:, ch])
    out[i0:i0 + n] *= np.linspace(1, 0.2, n)[:, None]
    return out


def master(mix, fade=0.05):
    mix = hp(mix.T, 25).T
    mix = np.tanh(mix * 0.9) / np.tanh(0.9)
    mix /= np.max(np.abs(mix)) / 0.93
    f = int(fade * SR); mix[-f:] *= np.linspace(1, 0, f)[:, None]
    return mix


# ---------- song material ----------
# Am - F - C - G  (vi IV I V), 2 beats each
PROG = [(57, [69, 72, 76, 79]), (53, [65, 69, 72, 76]), (48, [64, 67, 72, 76]), (55, [62, 67, 71, 74])]
HOOK = [(0, 76, .5, "a"), (.5, 76, .5, "o"), (1, 79, .5, "a"), (1.5, 81, .5, "e"),
        (2, 81, .5, "a"), (2.5, 79, .25, "o"), (2.75, 77, .25, "o"), (3, 76, 1, "a"),
        (4, 72, .5, "e"), (4.5, 76, .5, "a"), (5, 79, .5, "o"), (5.5, 84, .5, "i"),
        (6, 83, .5, "a"), (6.5, 79, .5, "o"), (7, 81, .5, "a"), (7.5, 83, .5, "e")]


def drop_section(L, start, beats=8, lead=True, energy=1.0):
    """Returns (chords, drums, bass, lead) buffers + kick times for a future-bass drop."""
    chords, drums, bass, leadb = (np.zeros(L) for _ in range(4))
    kicks = []
    for bar in range(beats // 4):
        b0 = start + bar * 4 * BEAT
        for kb in (0, 1.5, 2.75):
            kicks.append(b0 + kb * BEAT); place(drums, kick(), b0 + kb * BEAT, 1.0)
        place(drums, clap(), b0 + 2 * BEAT, 0.8 * energy)
        for s in range(16):  # 16th hats with a triplet roll at the end
            if s >= 14: continue
            place(drums, hat(), b0 + s * BEAT / 4, 0.6 if s % 2 else 0.9)
        for r in range(6):
            place(drums, hat(), b0 + 3.5 * BEAT + r * BEAT / 12, 0.4 + 0.08 * r)
    for i in range(beats // 2):
        root, notes = PROG[i % 4]; at = start + i * 2 * BEAT
        for nn in notes + [notes[0] + 12]:
            s = lp(supersaw(hz(nn), 2 * BEAT), 3500 * energy + 800)
            # future-bass 'wobble': 8th-note triplet gate swell
            tt = t(2 * BEAT); wob = 0.65 + 0.35 * np.sin(2 * np.pi * tt / (BEAT / 3)) ** 2
            place(chords, s * wob * adsr(len(s), a=0.01, d=0.2, s=0.85, r=0.04), at, 0.09)
        place(bass, bass808(root - 24, 2 * BEAT), at, 0.75)
    if lead:
        for beat, note, ln, v in HOOK[: beats * 2]:
            if beat >= beats: continue
            place(leadb, chop(note, ln * BEAT * 0.92, v), start + beat * BEAT, 0.32)
            place(leadb, pluck(note + 12, ln * BEAT), start + beat * BEAT, 0.08)
    return chords, drums, bass, leadb, kicks


def final_hit(bufs, at, tail=1.8):
    chords, drums, bass, leadb = bufs
    for nn in [60, 64, 67, 71, 74, 79]:   # Cmaj9 shimmer
        s = lp(supersaw(hz(nn), tail), 4000)
        place(chords, s * adsr(len(s), a=0.005, d=0.6, s=0.5, r=0.6), at, 0.1)
    place(bass, bass808(36, tail), at, 0.8)
    place(drums, kick(), at, 1.1)
    place(drums, clap(), at, 0.8)
    place(drums, impact(tail), at, 0.7)
    place(leadb, chop(84, 0.6, "i"), at, 0.35)
    place(leadb, chop(79, 0.6, "a"), at, 0.15)


def render(chords, drums, bass, leadb, fx, kicks):
    L = len(chords)
    sc = sidechain(L, kicks)
    music = chords * sc + leadb
    wet = reverb(music + 0.3 * fx, mix=0.35)
    low = (bass * sc ** 0.5)[:, None]
    mix = wet + low + drums[:, None] + 0.5 * np.stack([fx, fx], 1)
    # stereo width for chords
    d = int(0.012 * SR); width = np.zeros_like(chords); width[d:] = (chords * sc)[:-d]
    mix[:, 0] += 0.25 * chords * sc; mix[:, 1] += 0.25 * width
    return mix


# ---------- arrangements ----------
def full_jingle():
    intro, build = 4, 4
    drop_at = (intro) * BEAT + 0.15  # tiny breath before the drop
    L = int((drop_at + 8 * BEAT + 2.2) * SR)
    fx = np.zeros(L)
    # intro: filtered hook over a soft pad, rising
    pre = np.zeros(L)
    for beat, note, ln, v in HOOK[:8]:
        place(pre, chop(note, ln * BEAT * 0.9, v), beat * BEAT, 0.3)
    for i in range(2):
        root, notes = PROG[i]
        for nn in notes:
            place(pre, supersaw(hz(nn), 2 * BEAT) * adsr(int(2 * BEAT * SR), a=0.3, d=0.1, s=1, r=0.05), i * 2 * BEAT, 0.05)
    pre[: int(drop_at * SR)] = sweep_lp(pre[: int(drop_at * SR)], 400, 6000)
    pre[int((drop_at - 0.15) * SR):] = 0
    # build: snare roll accelerating + riser + reverse cymbal into the drop
    roll_start = 2 * BEAT
    for k, frac in enumerate([0, .5, 1, 1.5, 1.75, 1.875, 1.9375, 2.0 - 1 / 32]):
        if roll_start + frac * BEAT < drop_at - 0.15:
            place(fx, clap(0.15), roll_start + frac * BEAT, 0.25 + 0.1 * k)
    place(fx, noise_riser(drop_at - 0.15), 0, 0.5)
    rc = reverse_cymbal(1.2); place(fx, rc, drop_at - len(rc) / SR, 0.9)
    chords, drums, bass, leadb, kicks = drop_section(L, drop_at, beats=8)
    leadb += pre
    end_at = drop_at + 8 * BEAT
    final_hit((chords, drums, bass, leadb), end_at)
    kicks.append(end_at)
    mix = render(chords, drums, bass, leadb, fx, kicks)
    mix = tape_stop(mix, end_at + 0.55, 0.9)
    return master(mix)


def stinger():
    L = int(3.2 * SR); fx = np.zeros(L)
    chords, drums, bass, leadb = (np.zeros(L) for _ in range(4))
    for k, (n, v) in enumerate([(72, "a"), (76, "o"), (79, "e")]):
        place(leadb, chop(n, 0.16, v), k * 0.17, 0.3)
    hit = 0.55
    place(fx, reverse_cymbal(0.5), hit - 0.5, 0.8)
    final_hit((chords, drums, bass, leadb), hit, tail=2.4)
    for k, (n, v) in enumerate([(84, "i"), (83, "a"), (79, "o"), (84, "i")]):
        place(leadb, chop(n, 0.18, v), hit + 0.6 + k * BEAT / 2, 0.25)
    mix = render(chords, drums, bass, leadb, fx, [hit])
    mix = tape_stop(mix, hit + 1.6, 0.9)
    return master(mix)


def bed():
    beats = 32
    L = int((beats * BEAT + 1.5) * SR); fx = np.zeros(L)
    chords, drums, bass, leadb, kicks = drop_section(L, 0, beats=beats, lead=False, energy=0.55)
    # softer, darker: no claps-heavy, lowpassed chords so speech sits on top
    chords = lp(chords, 2200)
    mix = render(chords, drums * 0.7, bass * 0.8, leadb, fx, kicks)
    m = master(mix, fade=1.2)
    return m * 0.8


def save(path, x):
    data = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(data.tobytes())


if __name__ == "__main__":
    save("zeroboys_jingle.wav", full_jingle())
    save("zeroboys_jingle_short.wav", stinger())
    save("zeroboys_bed.wav", bed())
    print("done")
