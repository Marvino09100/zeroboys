"""Synthesizes an instrumental radio jingle (station ident) with numpy only.

Output: zeroboys_jingle.wav (full ident, ~7 s) and zeroboys_jingle_short.wav (~3 s stinger).
"""
import wave
import numpy as np

SR = 44100
BPM = 128
BEAT = 60 / BPM
rng = np.random.default_rng(7)


def t(dur):
    return np.arange(int(dur * SR)) / SR


def env(n, a=0.005, d=0.2, s=0.6, r=0.15):
    x = np.ones(n) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na = min(na, n); x[:na] = np.linspace(0, 1, na)
    nd = min(nd, n - na); x[na:na + nd] = np.linspace(1, s, nd)
    if nr < n: x[-nr:] *= np.linspace(1, 0, nr)
    return x


def hz(note):  # MIDI -> Hz
    return 440 * 2 ** ((note - 69) / 12)


def supersaw(f, dur, voices=7, detune=0.012):
    tt = t(dur); out = np.zeros_like(tt)
    for i in range(voices):
        df = f * (1 + detune * (i - voices // 2) / (voices // 2))
        ph = rng.random()
        out += 2 * ((tt * df + ph) % 1) - 1
    return out / voices


def lowpass(x, cutoff):
    # simple one-pole, cutoff may be array
    cutoff = np.broadcast_to(cutoff, x.shape)
    a = 1 - np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x); acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc); y[i] = acc
    return y


def kick(dur=0.45):
    tt = t(dur)
    f = 45 + 120 * np.exp(-tt * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-tt * 7) * 1.0


def snare(dur=0.3):
    tt = t(dur)
    noise = rng.standard_normal(len(tt)) * np.exp(-tt * 18)
    tone = np.sin(2 * np.pi * 190 * tt) * np.exp(-tt * 25)
    return 0.5 * noise + 0.4 * tone


def hat(dur=0.08):
    tt = t(dur)
    n = rng.standard_normal(len(tt))
    n = n - lowpass(n, 7000)
    return n * np.exp(-tt * 60) * 0.35


def crash(dur=2.5):
    tt = t(dur)
    n = rng.standard_normal(len(tt))
    n = n - lowpass(n, 4000)
    return n * np.exp(-tt * 1.8) * 0.3


def riser(dur):
    tt = t(dur)
    n = rng.standard_normal(len(tt))
    cut = 300 + 9000 * (tt / dur) ** 2
    return lowpass(n, cut) * (tt / dur) ** 2 * 0.6


def bell(f, dur=1.2):
    tt = t(dur)
    mod = np.sin(2 * np.pi * f * 3.5 * tt) * 2.0 * np.exp(-tt * 4)
    return np.sin(2 * np.pi * f * tt + mod) * np.exp(-tt * 3.5)


def place(buf, sig, at, gain=1.0):
    i = int(at * SR)
    end = min(len(buf), i + len(sig))
    buf[i:end] += sig[:end - i] * gain


def stereo_reverb(x, mix=0.25):
    out = np.zeros((len(x), 2))
    for ch, delays in enumerate([(0.029, 0.037, 0.053, 0.071), (0.031, 0.041, 0.059, 0.067)]):
        wet = np.zeros_like(x)
        for d in delays:
            n = int(d * SR); g = 0.55
            y = x.copy()
            for k in range(1, 8):
                sh = n * k
                if sh >= len(x): break
                y[sh:] += x[:-sh] * g ** k
            wet += y
        wet = lowpass(wet / len(delays), 6000)
        out[:, ch] = x + mix * wet
    return out


def build(full=True):
    if full:
        length = 7.5
        b = lambda n: 0.6 + n * BEAT  # musical start after riser
    else:
        length = 3.4
        b = lambda n: 0.0 + n * BEAT
    music = np.zeros(int(length * SR))
    drums = np.zeros_like(music)
    fx = np.zeros_like(music)

    # Signature motif: "ze-ro-boys ra-di-o" — C major lift into big Fsus/F -> C
    motif = [(0, 72, 0.5), (0.5, 76, 0.5), (1, 79, 0.5), (1.5, 84, 0.5), (2, 83, 0.5), (2.5, 84, 1.5)]
    if full:
        place(fx, riser(0.6 + 2 * BEAT), 0.0, 0.8)
        # intro: arpeggio motif with bells over pumping chords
        for beat, note, dur in motif:
            place(music, bell(hz(note), 1.0), b(beat), 0.35)
            place(music, bell(hz(note + 12), 0.6), b(beat), 0.08)
        chords = [(0, [60, 64, 67, 72], 2), (2, [57, 60, 64, 69], 2), (4, [53, 57, 60, 65], 2), (6, [55, 59, 62, 67], 2)]
        for beat, notes, dur in chords:
            for n in notes:
                s = supersaw(hz(n), dur * BEAT)
                s = lowpass(s, 900 + 2500 * np.linspace(0, 1, len(s)))
                place(music, s * env(len(s), d=0.3, s=0.7), b(beat), 0.14)
            bass = supersaw(hz(notes[0] - 24), dur * BEAT, voices=3, detune=0.004)
            place(music, lowpass(bass, 400) * env(len(bass)), b(beat), 0.5)
        for i in range(8):
            place(drums, kick(), b(i), 0.9)
            place(drums, hat(), b(i + 0.5), 1.0)
            if i % 2 == 1:
                place(drums, snare(), b(i), 0.7)
        # snare roll into the hit
        for k in range(8):
            place(drums, snare(0.15), b(7 + k / 8), 0.25 + 0.06 * k)
        place(fx, riser(2 * BEAT), b(6), 0.9)
        hit_at = b(8)
    else:
        hit_at = 0.0
        place(fx, riser(0.3), 0.0, 0.3)

    # Big final hit: C major with sus-ish sparkle
    final = [48, 60, 64, 67, 72, 76, 79]
    tail = 2.6
    for n in final:
        s = supersaw(hz(n), tail)
        s = lowpass(s, 5000 * np.exp(-t(tail) * 1.2) + 600)
        place(music, s * env(len(s), d=1.0, s=0.4, r=1.2), hit_at, 0.12 if n > 40 else 0.4)
    sub = np.sin(2 * np.pi * hz(36) * t(tail)) * np.exp(-t(tail) * 1.5)
    place(music, sub, hit_at, 0.5)
    for k, n in enumerate([84, 88, 91, 96]):
        place(music, bell(hz(n), 1.5), hit_at + 0.06 * k, 0.18)
    place(drums, kick(), hit_at, 1.2)
    place(drums, snare(0.5), hit_at, 0.8)
    place(drums, crash(), hit_at, 1.0)
    if not full:
        for beat, note, _ in motif[:4]:
            place(music, bell(hz(note + 12), 0.8), hit_at + 0.4 + beat * 0.25, 0.22)

    mix = stereo_reverb(music + 0.6 * fx, 0.3)
    mix += np.stack([drums, drums], 1)
    # gentle stereo widening for fx
    mix[:, 0] += fx * 0.15; mix[:, 1] -= fx * 0.05
    # soft-clip master + normalize
    mix = np.tanh(mix * 1.4)
    mix /= np.max(np.abs(mix)) / 0.95
    fade = int(0.3 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    return mix


def save(path, x):
    data = (x * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(data.tobytes())


if __name__ == "__main__":
    save("zeroboys_jingle.wav", build(True))
    save("zeroboys_jingle_short.wav", build(False))
    print("done")
