#!/usr/bin/env python
"""Capture a DJI Tello flight clip (+ telemetry) into data/flights/.

SAFETY: this tool does NOT fly the drone. It only records the video stream and
telemetry while a human pilot flies manually. Read the safety notes below and in
data/README.md before every session.

Examples::

    # Health check only (connect, print battery, disconnect -- no recording):
    python scripts/capture_tello.py --preflight

    # Record a 30s clip at 15 fps into data/flights/:
    python scripts/capture_tello.py --duration 30 --fps 15

    # Named session with tags and a higher battery floor:
    python scripts/capture_tello.py --duration 45 --name-prefix hover \
        --tags indoor calm --min-battery 30 --high-res

Output (timestamped):
    data/flights/<prefix>_<YYYYmmdd_HHMMSS>[_tags].mp4     video
    data/flights/<prefix>_<YYYYmmdd_HHMMSS>[_tags].csv     training actions
    data/flights/<prefix>_<YYYYmmdd_HHMMSS>[_tags].raw.csv full flight log
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

SAFETY_BANNER = """
============================ TELLO CAPTURE - SAFETY ============================
 * This tool records only. A HUMAN PILOT must fly the drone manually.
 * Fly in a large, open space; keep props clear of people, pets, and hands.
 * Check battery >= --min-battery. A low battery risks an uncontrolled landing.
 * Indoors: good lighting, no wind; watch for downwash near walls/ceilings.
 * Stop with Ctrl-C at any time - the video stream is closed cleanly on exit.
 * You are responsible for airspace/regulatory compliance (see data/README.md).
===============================================================================
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", default="data/flights")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to record.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--name-prefix", default="flight")
    parser.add_argument("--tags", nargs="*", default=[], help="Extra filename tags.")
    parser.add_argument("--min-battery", type=int, default=20, help="Abort below this %%.")
    parser.add_argument("--high-res", action="store_true", help="Request 720p stream.")
    parser.add_argument("--no-telemetry", action="store_true", help="Video only.")
    parser.add_argument(
        "--preflight", action="store_true",
        help="Connect, print battery/state, disconnect. No recording.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the safety confirmation.")
    args = parser.parse_args()

    print(SAFETY_BANNER)

    if args.preflight:
        from aerojepa.data.tello import preflight_check

        info = preflight_check(min_battery=args.min_battery)
        print("\n=== Tello preflight checklist ===")
        print(f"  opencv-python     : {'OK' if info.get('opencv') else 'MISSING - pip install opencv-python'}")
        print(f"  djitellopy        : {'OK' if info.get('djitellopy') else 'MISSING - pip install djitellopy'}")
        print(f"  Wi-Fi ({info.get('tello_host')}): {'OK' if info.get('wifi_reachable') else 'FAIL - join Tello Wi-Fi'}")
        if info.get("battery") is not None:
            print(f"  battery           : {info['battery']}% (min {info['min_battery']}%)")
            print(f"  temperature       : {info.get('temperature_c')} °C")
            print(f"  height            : {info.get('height_cm')} cm")
        print(f"\nOverall: {'PASS' if info.get('ok') else 'FAIL'}")
        sys.exit(0 if info.get("ok") else 1)

    if not args.yes:
        reply = input("Pilot ready and area clear? Type 'y' to record: ").strip().lower()
        if reply != "y":
            print("Aborted.")
            sys.exit(1)

    from aerojepa.data.tello import CaptureConfig, capture_flight

    config = CaptureConfig(
        out_dir=args.out_dir,
        duration_s=args.duration,
        fps=args.fps,
        name_prefix=args.name_prefix,
        tags=args.tags,
        min_battery=args.min_battery,
        high_res=args.high_res,
        record_telemetry=not args.no_telemetry,
    )
    result = capture_flight(config)
    print(f"\nDone. Video: {result.video_path}")
    if result.action_csv_path:
        print(f"Actions: {result.action_csv_path}")
        print(f"Raw log: {result.raw_csv_path}")
    print("\nNext: python scripts/preprocess_real.py --input-dir", args.out_dir)


if __name__ == "__main__":
    main()
