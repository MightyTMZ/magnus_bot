"""Manually jog a single arm's joints by small increments (live hardware helper).

SAFETY: this MOVES an arm. It refuses to run unless ALL of the following hold:
  * --live is passed,
  * config.measured is true,
  * --i-have-approval is passed (stands in for explicit in-session operator approval).

Even then, jogging is clamped to joint limits and a small per-command step. STUB (M1): the
live jog loop itself is wired once LeRobotArm is validated on hardware.
"""

from __future__ import annotations

import argparse
import sys

from magnus_bot.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jog one arm (LIVE, gated).")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("arm_id", help="which arm to jog (e.g. follower_1)")
    parser.add_argument("--live", action="store_true", help="actually command hardware")
    parser.add_argument("--i-have-approval", action="store_true",
                        help="confirm in-session operator approval to move the arm")
    args = parser.parse_args(argv)

    config = load_config(args.config_dir)
    if not args.live:
        print("Refusing: jog_arm moves hardware. Re-run with --live (and approval).", file=sys.stderr)
        return 1
    if not args.i_have_approval:
        print("Refusing: --live jog needs explicit operator approval (--i-have-approval).", file=sys.stderr)
        return 1
    try:
        config.require_measured()
    except RuntimeError as exc:
        print(f"Refusing: {exc}", file=sys.stderr)
        return 1
    # TODO(M1): construct LeRobotArm(arm_id, port), connect, then step joints within limits.
    print("jog loop not yet implemented (M1): validate LeRobotArm on hardware first.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
