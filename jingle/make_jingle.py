"""BuJuLa-Radio jingle package - urban / R&B radio style, original composition.

Real instrument samples come from the FluidR3 GM soundfont rendered with FluidSynth;
808, kick, claps, hats and FX are synthesized. Requirements:
    apt-get install fluidsynth fluid-soundfont-gm   &&   pip install numpy scipy mido

Outputs (WAV 44.1 kHz stereo, + MP3 via ffmpeg in build step):
  bujula_radio_jingle.wav   full ident (~7 s)
  bujula_radio_luecke.wav   same ident with a ~1.4 s break to drop a spoken/recorded
                            "BuJuLa-Radio!" into (break starts at LUECKE_START seconds)
  bujula_radio_stinger.wav  short hit (~3.5 s) between songs
  bujula_radio_bed.wav      music bed (~20 s) to talk over
"""
import os
import subprocess
import tempfile
import wave

import mido
import numpy as np
from scipy.signal import butter, sosfilt, lfilter

SR = 44100
BPM = 100
BEAT = 60 / BPM
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
rng = np.random.default_rng(3)

# GM programs (0-based)
RHODES, STRINGS, SLOW_STRINGS, BRASS, CELESTA, GLOCK, CHOIR, PAD, VIBES = 4, 48, 49, 61, 8, 9, 52, 89, 11


# ---------------------------------------------------------------- helpers
def t(dur):
    return np.arange(int(dur * SR)) / SR


def hz(n):
    return 440 * 2 ** ((n - 69) / 12)


def place(buf, sig, at, gain=1.0):
    i = int(round(at * SR))
    if i < 0: sig, i = sig[-i:], 0
    if i >= len(buf): return
    end = min(len(buf), i + len(sig))
    buf[i:end] += sig[:end - i] * gain


def filt(x, fc, kind, order=2):
    sos = butter(order, min(fc, SR / 2 - 100) / (SR / 2), kind, output="sos")
    return sosfilt(sos, x, axis=0)


def sweep_lp(x, f0, f1):
    out = np.zeros_like(x); blk = 1024; nb = len(x) // blk + 1; zi = None
    for k in range(nb):
        seg = x[k * blk:(k + 1) * blk]
        if not len(seg): break
        sos = butter(2, min(f0 * (f1 / f0) ** (k / max(1, nb - 1)), SR / 2 - 100) / (SR / 2), "low", output="sos")
        if zi is None: zi = np.zeros((sos.shape[0], 2))
        out[k * blk:k * blk + len(seg)], zi = sosfilt(sos, seg, zi=zi)
    return out


def stereo(x, pan=0.5):
    return np.stack([x * np.cos(pan * np.pi / 2), x * np.sin(pan * np.pi / 2)], 1) * np.sqrt(2)


# ---------------------------------------------------------------- soundfont stems
def render_midi(notes, program, length, gain=1.0, expression=None, channel=0):
    """notes: list of (beat, midi_note, length_beats, velocity). Returns stereo array of `length` samples."""
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM)))
    ev = [(0, mido.Message("program_change", program=program, channel=channel))]
    ev.append((0, mido.Message("control_change", control=91, value=70, channel=channel)))  # reverb send
    ev.append((0, mido.Message("control_change", control=93, value=25, channel=channel)))  # chorus send
    if expression:  # list of (beat, value) for CC11 swells
        ev += [(int(b * tpb), mido.Message("control_change", control=11, value=int(v), channel=channel)) for b, v in expression]
    for b, n, ln, vel in notes:
        ev.append((int(b * tpb), mido.Message("note_on", note=n, velocity=vel, channel=channel)))
        ev.append((int((b + ln) * tpb), mido.Message("note_off", note=n, velocity=0, channel=channel)))
    ev.sort(key=lambda e: (e[0], e[1].type == "note_on"))
    last = 0
    for tick, msg in ev:
        tr.append(msg.copy(time=tick - last)); last = tick
    tr.append(mido.MetaMessage("end_of_track", time=tpb * 8))
    with tempfile.TemporaryDirectory() as d:
        mp, wp = os.path.join(d, "s.mid"), os.path.join(d, "s.wav")
        mid.save(mp)
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.8", "-r", str(SR), "-F", wp, SF2, mp],
                       check=True, capture_output=True)
        with wave.open(wp) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2) / 32768
    out = np.zeros((length, 2)); n = min(length, len(x)); out[:n] = x[:n]
    return out * gain


# ---------------------------------------------------------------- synthesized drums / fx
def kick(dur=0.4):
    tt = t(dur)
    f = 48 + 150 * np.exp(-tt * 40)
    k = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 8)
    click = filt(rng.standard_normal(len(tt)), 2500, "high") * np.exp(-tt * 400) * 0.4
    return np.tanh(1.8 * (k + click))


def bass808(note, dur, glide_from=None):
    tt = t(dur)
    f = hz(note) * np.ones(len(tt))
    if glide_from is not None:
        f = hz(note) + (hz(glide_from) - hz(note)) * np.exp(-tt * 25)
    f *= 1 + 0.8 * np.exp(-tt * 40)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR)
    s = np.tanh(2.2 * s)  # harmonics so the 808 is audible on small speakers
    e = np.minimum(1, tt / 0.003) * np.exp(-tt * 1.2)
    rel = int(0.04 * SR); e[-rel:] *= np.linspace(1, 0, rel)
    return s * e


def clap(dur=0.4):
    tt = t(dur); n = len(tt)
    noise = filt(filt(rng.standard_normal(n), 900, "high"), 5000, "low")
    e = np.zeros(n)
    for off, dec in ((0, 90), (0.009, 90), (0.019, 90), (0.028, 16)):
        i = int(off * SR); e[i:] = np.maximum(e[i:], np.exp(-tt[:n - i] * dec))
    body = np.sin(2 * np.pi * 200 * tt) * np.exp(-tt * 30) * 0.3
    return (noise * e + body) * 1.3


def snare(dur=0.25):
    tt = t(dur)
    nz = filt(rng.standard_normal(len(tt)), 1800, "high") * np.exp(-tt * 22)
    tone = np.sin(2 * np.pi * np.cumsum(220 + 60 * np.exp(-tt * 50)) / SR) * np.exp(-tt * 30)
    return 0.6 * nz + 0.5 * tone


def hat(open_=False):
    tt = t(0.3 if open_ else 0.06)
    # metallic: sum of square-ish partials, then highpass
    f0 = [205.3, 304.4, 369.6, 522.7, 540, 800]
    m = sum(np.sign(np.sin(2 * np.pi * f * tt)) for f in f0)
    m = filt(m + 0.5 * rng.standard_normal(len(tt)), 7500, "high", 4)
    return m * np.exp(-tt * (9 if open_ else 70)) * 0.12


def reverse_cymbal(dur):
    tt = t(dur)
    n = filt(rng.standard_normal(len(tt)), 4500, "high") * np.exp(-tt * 3.5)
    return n[::-1] * np.linspace(0, 1, len(n)) ** 1.5 * 0.5


def crash(dur=3.0):
    tt = t(dur)
    f0 = [340, 517, 733, 1043, 1311, 1690]
    m = sum(np.sin(2 * np.pi * f * tt + rng.random() * 6) for f in f0) * 0.1
    n = filt(rng.standard_normal(len(tt)) + m, 3500, "high")
    return n * np.exp(-tt * 1.6) * 0.3


def impact(dur=2.5):
    tt = t(dur)
    boom = np.sin(2 * np.pi * np.cumsum(38 + 70 * np.exp(-tt * 12)) / SR) * np.exp(-tt * 2.2)
    return np.tanh(1.5 * boom)


def riser(dur):
    n = rng.standard_normal(int(dur * SR))
    return sweep_lp(n, 300, 9000) * np.linspace(0, 1, len(n)) ** 2.5 * 0.5


def downlifter(dur):
    n = rng.standard_normal(int(dur * SR))
    return sweep_lp(n, 8000, 200) * np.linspace(1, 0, len(n)) ** 2 * 0.4


# ---------------------------------------------------------------- mix utilities
def sidechain(length, times, depth=0.5, release=0.18):
    g = np.ones(length)
    curve = 1 - depth * np.exp(-np.arange(int(release * 4 * SR)) / (release * SR))
    for kt in times:
        i = int(kt * SR); e = min(length, i + len(curve))
        if i < length: g[i:e] = np.minimum(g[i:e], curve[:e - i])
    return g[:, None]


def plate(x, mix=0.2):
    """Small stereo plate-style reverb for the synthesized drums."""
    m = x.mean(1) if x.ndim == 2 else x
    out = np.zeros((len(m), 2))
    for ch, ds in enumerate([(0.0253, 0.0269, 0.0293, 0.0307), (0.0261, 0.0277, 0.0301, 0.0317)]):
        w = sum(lfilter([1], np.r_[1, np.zeros(int(d * SR) - 1), -0.72], m) for d in ds) / 4
        out[:, ch] = filt(filt(w, 300, "high"), 6000, "low")
    return (x if x.ndim == 2 else stereo(x)) + mix * out


def compress(x, thresh_db=-14, ratio=3.0, attack=0.005, release=0.12):
    lvl = np.abs(x).max(1)
    a_a, a_r = np.exp(-1 / (attack * SR)), np.exp(-1 / (release * SR))
    # envelope follower (vectorised approx: separate attack/release passes)
    env = lfilter([1 - a_r], [1, -a_r], lvl)
    env = np.maximum(env, lfilter([1 - a_a], [1, -a_a], lvl))
    db = 20 * np.log10(env + 1e-9)
    over = np.maximum(0, db - thresh_db)
    gain = 10 ** (-over * (1 - 1 / ratio) / 20)
    return x * gain[:, None]


def master(x, fade=0.08, ceiling=0.93):
    x = filt(x, 28, "high")
    # gentle "smile" EQ: a little extra air and low end
    x = x + 0.18 * filt(x, 9000, "high") + 0.12 * filt(x, 90, "low")
    x = compress(x / (np.max(np.abs(x)) + 1e-9), -20, 4.0)
    x = compress(x / (np.max(np.abs(x)) + 1e-9), -8, 10.0, attack=0.001, release=0.05)  # peak limiting
    x = np.tanh(x * 1.8) / np.tanh(1.8)
    x *= ceiling / np.max(np.abs(x))
    f = int(fade * SR); x[-f:] *= np.linspace(1, 0, f)[:, None]
    return x


# ---------------------------------------------------------------- music (key: Ab major)
# chords as (bass note for 808, voicing)
DBMAJ9 = (37, [53, 60, 65, 68, 72])      # Db  F C F Ab C   -> Dbmaj9 sound
EB11 = (39, [51, 61, 65, 68, 70])        # Eb  Db F Ab Bb  -> Eb11
CM7 = (36, [55, 63, 67, 70])             # C   G Eb G Bb   -> Cm7
FM9 = (41, [56, 63, 67, 68, 72])         # F   Ab Eb G Ab C -> Fm9
ABMAJ9 = (32, [56, 60, 63, 67, 70, 75])  # Ab  Ab C Eb G Bb Eb

# Signature motif ("Bu-Ju-La Ra-di-o" rhythm), relative beats
MOTIF = [(0.5, 75, .25), (0.75, 77, .25), (1.0, 80, .5), (2.0, 84, .5), (2.5, 82, .5), (3.0, 80, 1.0)]


def groove(L, start, bars, chords_per_bar, intensity=1.0, fill_last=True):
    """Drums + 808 + Rhodes for `bars` bars starting at time `start` (s). Returns stems dict."""
    drums = np.zeros(L); hats = np.zeros(L); b808 = np.zeros(L)
    rh_notes, kicks = [], []
    beat0 = start / BEAT
    for bar in range(bars):
        bb = bar * 4
        last = fill_last and bar == bars - 1
        for k in (0, 0.75, 2.5) + (() if last else (3.75,)):
            kicks.append(start + (bb + k) * BEAT); place(drums, kick(), start + (bb + k) * BEAT, 0.95)
        for s_ in (1, 3):
            place(drums, clap(), start + (bb + s_) * BEAT, 0.55 * intensity)
            place(drums, snare(), start + (bb + s_) * BEAT, 0.35 * intensity)
        # hats: 8ths with 16th/triplet rolls (trap-style)
        steps = [i * 0.5 for i in range(8)]
        for s_ in steps:
            if last and s_ >= 3.5: continue
            place(hats, hat(), start + (bb + s_) * BEAT, 0.9 if s_ % 1 == 0 else 0.6)
        for r in range(4):
            place(hats, hat(), start + (bb + 1.5 + r * 0.125) * BEAT, 0.4 + 0.1 * r)
        if not last:
            for r in range(6):
                place(hats, hat(), start + (bb + 3.5 + r / 12) * BEAT, 0.35 + 0.08 * r)
        else:  # snare fill into the hit
            for r in range(4):
                place(drums, snare(0.15), start + (bb + 3.5 + r * 0.125) * BEAT, 0.25 + 0.12 * r)
        place(hats, hat(True), start + (bb + 2) * BEAT, 0.5)
        for c_i, (root, voicing) in enumerate(chords_per_bar[bar]):
            n_ch = len(chords_per_bar[bar]); cb = bb + c_i * 4 / n_ch; ln = 4 / n_ch
            if last and c_i == n_ch - 1: ln -= 0.5
            prev = chords_per_bar[bar][c_i - 1][0] if c_i else None
            place(b808, bass808(root - 12, ln * BEAT * 0.98, glide_from=(prev - 12) if prev else None),
                  start + cb * BEAT, 0.8)
            # Rhodes: chord on the beat + rhythmic re-hit (R&B comping)
            for off, l2, vel in ((0, 1.4, 92), (1.5, 0.4, 70)):
                if off < ln:
                    rh_notes += [(beat0 + cb + off, n, min(l2, ln - off), vel) for n in voicing]
    return drums, hats, b808, rh_notes, kicks


def final_hit(L, at, tail_beats=5):
    """Big tonic hit: strings + brass stab + choir pad + celesta cascade + impact."""
    b = at / BEAT
    root, voicing = ABMAJ9
    stems = {}
    stems["strings"] = render_midi([(b, n, tail_beats * 0.7, 110) for n in voicing + [79, 84, 87]],
                                   STRINGS, L, 0.9)
    stems["brass"] = render_midi([(b, n, 0.6, 115) for n in (68, 72, 75, 79)], BRASS, L, 0.55)
    stems["choir"] = render_midi([(b, n, tail_beats * 0.6, 85) for n in (63, 68, 72, 75)], CHOIR, L, 0.35)
    casc = [87, 84, 80, 79, 75, 72, 80, 84, 87, 92]
    stems["celesta"] = render_midi([(b + 0.08 + i * 0.125, n, 1.5, 100 - i * 4) for i, n in enumerate(casc)],
                                   CELESTA, L, 0.55)
    stems["rhodes_hit"] = render_midi([(b, n, 2.5, 105) for n in voicing], RHODES, L, 0.8)
    fx = np.zeros(L); place(fx, impact(), at, 0.8)
    drums = np.zeros(L); place(drums, kick(), at, 1.0); place(drums, clap(), at, 0.7)
    cr = np.zeros(L); place(cr, crash(), at, 1.0)
    b808 = np.zeros(L); place(b808, bass808(root, tail_beats * BEAT * 0.8), at, 0.85)
    return stems, fx, drums, cr, b808


def ident(luecke=False):
    """Full jingle. Timeline (beats): 0-1 swell, 1-9 two bars groove, hit."""
    intro = 1.0
    groove_start = intro * BEAT
    gap = 2.4 if luecke else 0.5        # beats of break before the hit
    hit = groove_start + 8 * BEAT + (gap - 0.5) * BEAT
    L = int((hit + 5.2 * BEAT + 0.3) * SR)
    drums, hats, b808, rh, kicks = groove(L, groove_start, 2, [[DBMAJ9, EB11], [CM7, FM9]])
    gb = groove_start / BEAT
    # signature motif on vibes + celesta (octave), bar 1 and answer in bar 2
    mot = [(gb + b, n, ln, 100) for b, n, ln in MOTIF]
    ans = [(gb + 4 + b, n - (2 if n == 84 else 0), ln, 95) for b, n, ln in MOTIF[:5]] + [(gb + 7.0, 79, 0.5, 95)]
    lead = render_midi(mot + ans, VIBES, L, 0.75) + render_midi([(b, n + 12, ln, 70) for b, n, ln, _ in mot + ans], CELESTA, L, 0.3)
    rhodes = render_midi(rh, RHODES, L, 0.85)
    # intro swell: strings crescendo on Db + reverse cymbal + riser
    swell = render_midi([(0, n, intro + 0.1, 100) for n in (53, 60, 65, 68, 72)], SLOW_STRINGS, L, 0.8,
                        expression=[(i / 16 * intro, 20 + 107 * i / 16) for i in range(17)])
    pad = render_midi([(gb, n, 7.5, 60) for n in (65, 68, 72)] + [(gb + 4, n, 3.5, 60) for n in (63, 67, 70)], PAD, L, 0.3)
    fx = np.zeros(L)
    rc = reverse_cymbal(intro * BEAT); place(fx, rc, 0, 0.9)
    place(fx, riser(3.5 * BEAT), groove_start + 4.5 * BEAT, 0.35)
    rc2 = reverse_cymbal(0.9); place(fx, rc2, hit - 0.9, 0.8)
    if luecke:
        # keep a soft pad + downlifter under the break so it doesn't sound like dead air
        place(fx, downlifter(1.2), groove_start + 8 * BEAT - 0.5 * BEAT, 0.6)
        pad += render_midi([(hit / BEAT - gap, n, gap - 0.1, 55) for n in (63, 68, 72)], PAD, L, 0.7)
    fh, hit_fx, hit_drums, cr, hit_808 = final_hit(L, hit)
    kicks.append(hit)
    sc = sidechain(L, kicks, 0.45)
    music = (rhodes + pad + swell) * sc + lead * (0.6 + 0.4 * sc) + sum(fh.values())
    beat = plate(stereo(drums + hit_drums), 0.25) + stereo(hats, 0.62) + stereo(cr, 0.4)
    low = stereo(b808 + hit_808) * 0.95
    mix = music + beat + low + stereo(fx, 0.5) + stereo(hit_fx) * 0.8
    return master(mix), (groove_start + 8 * BEAT - 0.25 * BEAT if luecke else None)


def stinger():
    L = int(3.6 * SR)
    hit = 0.75 * BEAT + 0.12
    # quick pickup: vibes motif tail + snare flam
    pick = render_midi([(0.0, 75, .25, 95), (0.25, 77, .25, 100), (0.5, 80, .25, 105)], VIBES, L, 0.8)
    fh, fx, drums, cr, b808 = final_hit(L, hit, tail_beats=5)
    place(drums, snare(0.15), hit - 0.25 * BEAT, 0.4); place(drums, snare(0.15), hit - 0.125 * BEAT, 0.55)
    rc = np.zeros(L); place(rc, reverse_cymbal(hit), 0, 0.8)
    mix = pick + sum(fh.values()) + plate(stereo(drums), 0.25) + stereo(cr, 0.4) + stereo(b808) * 0.95 \
        + stereo(fx) * 0.8 + stereo(rc, 0.5)
    return master(mix)


def bed():
    bars = 8
    L = int((bars * 4 * BEAT + 2.5) * SR)
    prog = [[DBMAJ9, EB11], [CM7, FM9]] * (bars // 2)
    drums, hats, b808, rh, kicks = groove(L, 0, bars, prog, intensity=0.7, fill_last=False)
    rhodes = render_midi(rh, RHODES, L, 0.75)
    pad_notes = []
    for bar in range(bars):
        for c_i, (root, voicing) in enumerate(prog[bar]):
            pad_notes += [(bar * 4 + c_i * 2, n, 2, 50) for n in voicing[-3:]]
    pad = render_midi(pad_notes, PAD, L, 0.3)
    sc = sidechain(L, kicks, 0.4)
    # no lead melody, mids slightly scooped so speech sits on top
    music = (rhodes + pad) * sc
    music = music - 0.3 * filt(filt(music, 800, "high"), 3000, "low")
    mix = music + plate(stereo(drums * 0.8), 0.2) + stereo(hats * 0.7, 0.62) + stereo(b808) * 0.85
    return master(mix, fade=2.0) * 0.85


def save(path, x):
    data = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(data.tobytes())


if __name__ == "__main__":
    j, _ = ident(False); save("bujula_radio_jingle.wav", j)
    l, gap_at = ident(True); save("bujula_radio_luecke.wav", l)
    save("bujula_radio_stinger.wav", stinger())
    save("bujula_radio_bed.wav", bed())
    print(f"done; Luecke beginnt bei {gap_at:.2f} s")
