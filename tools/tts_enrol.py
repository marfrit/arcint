#!/usr/bin/env python3
"""Enrol a speaker profile for the TTS service (`docs/milestone-1.2.0.md` D5).

The voice-cloning rule: **only enrolled voices**. The API never takes a
reference audio; a cloned voice can only be made from a profile that already
sits on the card host, in the server-side profile directory, **outside this
repository**. This tool is the enrolment route the API refuses to be: it
validates a reference recording and writes the profile.

A profile is one directory per name:

    <profiles>/<name>/reference.wav      the recording (10-30 s, mono/stereo WAV)
    <profiles>/<name>/reference.txt      its transcript (for clone alignment)
    <profiles>/<name>/profile.json       {"name", "language", "source", "licence", "enabled"}

Writing a real voice's recording into the repository, a log or a doc is
forbidden (`docs/milestone-1.2.0.md` D5). This script prints paths and
metadata only; it never echoes waveform data, and the profile directory is
chosen by the operator (`--profiles`, or `$ARCINT_TTS_PROFILES`), never
assumed from the working directory.

Exit status: 0 on success, 1 on a validation refusal, 2 on a usage error.
"""
import argparse
import json
import os
import shutil
import sys
import wave

MIN_SECONDS = 10.0   # the operator's 10-30 s envelope (D5)
MAX_SECONDS = 30.0
# A clone can work from less in principle; below the envelope it is a warning,
# not a refusal, so a deliberate short sample is not blocked.
WARN_SECONDS = 8.0


def wav_info(path):
    with wave.open(path, "rb") as w:
        fr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        width = w.getsampwidth()
    return {"sample_rate": fr, "frames": n, "channels": ch, "width_bytes": width,
            "seconds": (n / fr) if fr else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="the voice's name, [a-z0-9_-]")
    ap.add_argument("--audio", required=True, help="the reference recording (WAV)")
    ap.add_argument("--text", required=True, help="the recording's transcript file")
    ap.add_argument("--profiles", required=False,
                    help="the server-side profile directory (outside the repository); "
                         "default $ARCINT_TTS_PROFILES")
    ap.add_argument("--language", default="de", help="the profile's language")
    ap.add_argument("--source", default="", help="where the sample came from (provenance)")
    ap.add_argument("--licence", default="", help="the sample's licence when it is not your own voice")
    ap.add_argument("--force", action="store_true", help="replace an existing profile")
    args = ap.parse_args()

    profiles = args.profiles or os.environ.get("ARCINT_TTS_PROFILES", "")
    if not profiles:
        print("--profiles (or $ARCINT_TTS_PROFILES) is required: the profile directory is "
              "operator-local and outside this repository", file=sys.stderr)
        return 2
    name = args.name.strip()
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        print("--name must be [a-z0-9_-]", file=sys.stderr)
        return 2
    if not os.path.exists(args.audio):
        print("--audio not found: %s" % args.audio, file=sys.stderr)
        return 2
    if not os.path.exists(args.text):
        print("--text not found: %s" % args.text, file=sys.stderr)
        return 2
    if not "../" not in os.path.abspath(profiles) and profiles.rstrip("/").endswith((".", "..")):
        print("refusing a suspicious --profiles path", file=sys.stderr)
        return 2

    try:
        info = wav_info(args.audio)
    except wave.Error as e:
        print("--audio is not a readable WAV (%s); the carrier form for a profile is WAV" % e,
              file=sys.stderr)
        return 1
    if info["seconds"] < WARN_SECONDS:
        print("warning: the sample is %.1f s, shorter than the 10-30 s envelope" % info["seconds"],
              file=sys.stderr)
    if info["seconds"] > MAX_SECONDS + 5.0:
        print("the sample is %.1f s, longer than the 30 s envelope" % info["seconds"], file=sys.stderr)
        return 1
    if info["width_bytes"] not in (1, 2, 4):
        print("unsupported sample width (%d bytes)" % info["width_bytes"], file=sys.stderr)
        return 1

    dest = os.path.join(profiles, name)
    if os.path.exists(dest) and not args.force:
        print("profile '%s' exists; --force to replace (and the API's listeners see the change "
              "only after the TTS unit's enrolment reload)" % name, file=sys.stderr)
        return 1
    os.makedirs(dest, exist_ok=True)
    shutil.copyfile(args.audio, os.path.join(dest, "reference.wav"))
    with open(os.path.join(dest, "reference.txt"), "w") as f:
        f.write(open(args.text).read().strip() + "\n")
    meta = {"name": name, "language": args.language, "source": args.source,
            "licence": args.licence, "enabled": True,
            "seconds": round(info["seconds"], 3), "sample_rate": info["sample_rate"],
            "channels": info["channels"]}
    json.dump(meta, open(os.path.join(dest, "profile.json"), "w"), indent=1)

    print("enrolled '%s': %s (%.1f s, %d Hz, %d ch); the API serves it as a voice name only"
          % (name, dest, info["seconds"], info["sample_rate"], info["channels"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
