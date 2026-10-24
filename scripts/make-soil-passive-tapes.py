#!/usr/bin/env python3
"""Make the two recorded, motor-off soil controls without editing the source tape."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path, help="Fresh directory; existing data is never replaced")
    args = parser.parse_args()
    original = json.loads(gzip.decompress(args.fixture.read_bytes()))
    if original["version"] != 1:
        raise ValueError("Unsupported command tape")
    source = json.loads(original["initial_session_json"])
    if source["machine"].get("articulation") is None:
        raise ValueError("An articulated bank fixture is required")
    args.output.mkdir()
    for case in ("passive-springs", "free-hinges"):
        initial = json.loads(json.dumps(source))
        initial["machine"]["engine_on"] = False
        if case == "free-hinges":
            initial["machine"]["articulation"]["stiffness_nm_rad"] = [0.0] * 4
            initial["machine"]["articulation"]["damping_nms_rad"] = [0.0] * 4
        commands = [
            dict(drive=0.0, steer=0.0, swing=0.0, boom=0.0, stick=0.0, curl=0.0,
                 cut=False, dump=False)
            for _ in range(120)
        ]
        tape = dict(version=1, initial_session_json=json.dumps(initial), commands=commands)
        data = json.dumps(tape).encode()
        path = args.output / f"{case}.commands.json.gz"
        path.write_bytes(gzip.compress(data, mtime=0))
        print(path)


if __name__ == "__main__":
    main()
