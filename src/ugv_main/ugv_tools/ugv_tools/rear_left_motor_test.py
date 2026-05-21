#!/usr/bin/env python3
import argparse
import json
import threading
import time

import serial


def clamp_pwm(value):
    return max(-255, min(255, int(value)))


class SerialReader(threading.Thread):
    def __init__(self, ser):
        super().__init__(daemon=True)
        self.ser = ser
        self.running = True

    def run(self):
        while self.running:
            try:
                line = self.ser.readline()
            except serial.SerialException:
                return
            if line:
                try:
                    print(f"feedback: {line.decode('utf-8', errors='replace').rstrip()}")
                except Exception:
                    print(f"feedback bytes: {line!r}")

    def stop(self):
        self.running = False


def send_json(ser, payload, echo=False):
    data = json.dumps(payload, separators=(",", ":")) + "\n"
    if echo:
        print(f"send: {data.rstrip()}")
    ser.write(data.encode("utf-8"))
    ser.flush()


def send_stop(ser, echo=False):
    # Stop both the direct PWM path and the normal ROS velocity path.
    send_json(ser, {"T": 11, "L": 0, "R": 0}, echo)
    send_json(ser, {"T": 13, "X": 0, "Z": 0}, echo)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Slowly drive the rover's left-side motor output for rear-left motor diagnosis. "
            "The Waveshare controller exposes left/right side PWM, not separate rear-left PWM."
        )
    )
    parser.add_argument("--port", default="/dev/ttyAMA0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--pwm",
        type=int,
        default=70,
        help="Left-side PWM command, -255..255. Increase slowly if the motor does not start.",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=10.0, help="Command repeat rate in Hz.")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Invert the PWM sign for the test.",
    )
    parser.add_argument(
        "--read-feedback",
        action="store_true",
        help="Print serial feedback while sending commands.",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Print each JSON command sent.",
    )
    return parser


def main(args=None):
    parsed = build_parser().parse_args(args)
    pwm = clamp_pwm(parsed.pwm)
    if parsed.reverse:
        pwm = -pwm

    if parsed.rate <= 0:
        raise ValueError("--rate must be greater than 0")
    if parsed.duration <= 0:
        raise ValueError("--duration must be greater than 0")

    print("Rear-left motor test")
    print("Controller limitation: this sends left-side PWM only: {'T':11,'L':pwm,'R':0}.")
    print("For a true rear-left-only test, disconnect the front-left motor or connect only rear-left.")
    print("Make sure the rover is lifted, wheels are clear, and normal bringup is stopped.")
    print(f"Port={parsed.port} baud={parsed.baud} pwm={pwm} duration={parsed.duration:.1f}s")

    reader = None
    with serial.Serial(parsed.port, parsed.baud, timeout=0.1, dsrdtr=None) as ser:
        ser.setRTS(False)
        ser.setDTR(False)

        if parsed.read_feedback:
            reader = SerialReader(ser)
            reader.start()

        try:
            send_stop(ser, parsed.echo)
            time.sleep(0.25)
            deadline = time.monotonic() + parsed.duration
            period = 1.0 / parsed.rate

            while time.monotonic() < deadline:
                send_json(ser, {"T": 11, "L": pwm, "R": 0}, parsed.echo)
                time.sleep(period)
        except KeyboardInterrupt:
            print("Interrupted; stopping motors.")
        finally:
            send_stop(ser, parsed.echo)
            time.sleep(0.1)
            if reader is not None:
                reader.stop()

    print("Done. Sent stop commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
