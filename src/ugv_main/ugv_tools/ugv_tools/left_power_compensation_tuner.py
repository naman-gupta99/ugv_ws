#!/usr/bin/env python3
import argparse
import json
import time

import serial


def clamp_pwm(value):
    return max(-255, min(255, int(round(value))))


def send_json(ser, payload, echo=False):
    data = json.dumps(payload, separators=(",", ":")) + "\n"
    if echo:
        print(f"send: {data.rstrip()}")
    ser.write(data.encode("utf-8"))
    ser.flush()


def stop(ser, echo=False):
    send_json(ser, {"T": 11, "L": 0, "R": 0}, echo)
    send_json(ser, {"T": 13, "X": 0, "Z": 0}, echo)


def run_straight_test(ser, base_pwm, compensation, duration, rate, echo=False):
    left_pwm = clamp_pwm(base_pwm + compensation)
    right_pwm = clamp_pwm(base_pwm)
    period = 1.0 / rate
    deadline = time.monotonic() + duration

    print(
        f"Running {duration:.1f}s: base={base_pwm}, compensation={compensation}, "
        f"L={left_pwm}, R={right_pwm}"
    )

    while time.monotonic() < deadline:
        send_json(ser, {"T": 11, "L": left_pwm, "R": right_pwm}, echo)
        time.sleep(period)

    stop(ser, echo)
    print("Stopped.")


def parse_compensation(raw):
    text = raw.strip().lower()
    if text in {"q", "quit", "exit", "x"}:
        return None
    return int(round(float(text)))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Interactively tune left-side PWM compensation for straight-line testing."
    )
    parser.add_argument("--port", default="/dev/ttyAMA0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--base-pwm",
        type=int,
        default=80,
        help="Base PWM for the right side. Left side receives base + compensation.",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--echo", action="store_true", help="Print every JSON command.")
    return parser


def main(args=None):
    parsed = build_parser().parse_args(args)

    if parsed.rate <= 0:
        raise ValueError("--rate must be greater than 0")
    if parsed.duration <= 0:
        raise ValueError("--duration must be greater than 0")

    base_pwm = clamp_pwm(parsed.base_pwm)

    print("Left power compensation tuner")
    print("Stop normal rover bringup first so this script can own /dev/ttyAMA0.")
    print("Use a clear test area. Press Ctrl+C or enter q to exit.")
    print("Compensation is added to the left PWM: L = base + compensation, R = base.")
    print(f"Port={parsed.port} baud={parsed.baud} base_pwm={base_pwm}")

    with serial.Serial(parsed.port, parsed.baud, timeout=0.1, dsrdtr=None) as ser:
        ser.setRTS(False)
        ser.setDTR(False)
        stop(ser, parsed.echo)

        try:
            while True:
                raw = input("\nEnter left compensation PWM (example 0, 10, 20, -10) or q: ")
                try:
                    compensation = parse_compensation(raw)
                except ValueError:
                    print("Please enter a number, or q to exit.")
                    continue

                if compensation is None:
                    break

                run_straight_test(
                    ser,
                    base_pwm=base_pwm,
                    compensation=compensation,
                    duration=parsed.duration,
                    rate=parsed.rate,
                    echo=parsed.echo,
                )
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            stop(ser, parsed.echo)

    print("Done. Sent stop commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
