#!/usr/bin/env python3
"""Erzeugt Sprachdateien mit OpenAI TTS aus einer Textliste.

Eingabe: tts/texte.txt, eine Zeile pro Datei im Format
    dateiname: Text, der gesprochen werden soll
Leere Zeilen und Zeilen mit # werden ignoriert.

Aufruf:
    OPENAI_API_KEY=... python3 tts/generate.py [--voice coral] [--format mp3]
"""
import argparse
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
API_URL = "https://api.openai.com/v1/audio/speech"
DEFAULT_INSTRUCTIONS = (
    "Sprich natürliches, lebendiges Hochdeutsch wie ein echter Mensch: "
    "lockere Betonung, natürliche Pausen, nicht roboterhaft."
)


def parse_lines(path):
    for nr, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            sys.exit(f"{path}:{nr}: erwartet 'dateiname: Text'")
        name, text = line.split(":", 1)
        name = re.sub(r"[^\w\-]+", "_", name.strip()).strip("_")
        yield name, text.strip()


def synthesize(text, voice, fmt, model, instructions, speed):
    body = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": fmt,
        "speed": speed,
    }
    if instructions and model.startswith("gpt-4o"):
        body["instructions"] = instructions
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'proxy-injected')}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"API-Fehler {e.code}: {e.read().decode(errors='replace')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=HERE / "texte.txt", type=pathlib.Path)
    p.add_argument("--out", default=HERE / "audio", type=pathlib.Path)
    p.add_argument("--voice", default="coral",
                   help="z. B. coral, sage, shimmer, nova (weiblich); ash, onyx, echo, ballad (männlich)")
    p.add_argument("--model", default="gpt-4o-mini-tts")
    p.add_argument("--format", default="mp3", choices=["mp3", "wav", "opus", "aac", "flac"])
    p.add_argument("--speed", default=1.0, type=float)
    p.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in parse_lines(args.input):
        target = args.out / f"{name}.{args.format}"
        target.write_bytes(
            synthesize(text, args.voice, args.format, args.model, args.instructions, args.speed)
        )
        print(f"✓ {target.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
