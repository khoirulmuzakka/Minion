import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename Min_EV files: change prefix and shift function numbers by -1 for F>=3."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Target directory containing *_F*_Min_EV.txt files (default: current directory).",
    )
    parser.add_argument(
        "--old-prefix",
        default="arrde",
        help="Existing prefix to replace (default: arrde).",
    )
    parser.add_argument(
        "--new-prefix",
        required=True,
        help="New prefix to apply.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames without changing files.",
    )
    return parser.parse_args()


def is_min_ev_file(path: Path, prefix: str) -> bool:
    name = path.name
    return name.startswith(f"{prefix}_F") and name.endswith("_Min_EV.txt")


def parse_func_number(name: str, prefix: str) -> int:
    # Expected format: <prefix>_F<num>_Min_EV.txt
    start = len(prefix) + 2  # len("<prefix>_F")
    end = name.rfind("_Min_EV.txt")
    return int(name[start:end])


def main() -> None:
    args = parse_args()
    directory = Path(args.directory)
    old_prefix = args.old_prefix
    new_prefix = args.new_prefix

    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    files = [p for p in directory.iterdir() if p.is_file() and is_min_ev_file(p, old_prefix)]

    if not files:
        raise SystemExit(f"No files found with prefix '{old_prefix}' in {directory}")

    # Step 1: rename prefix old -> new using temporary names to avoid collisions.
    temp_map = {}
    for path in files:
        temp = path.with_name(path.name + ".tmp_rename")
        temp_map[path] = temp

    for src, tmp in temp_map.items():
        if args.dry_run:
            print(f"{src.name} -> {tmp.name}")
        else:
            src.rename(tmp)

    # Step 2: rename temp files to new prefix.
    renamed = []
    for src, tmp in temp_map.items():
        new_name = tmp.name.replace(old_prefix + "_", new_prefix + "_", 1)
        dst = tmp.with_name(new_name.replace(".tmp_rename", ""))
        renamed.append(dst)
        if args.dry_run:
            print(f"{tmp.name} -> {dst.name}")
        else:
            tmp.rename(dst)

    # Step 3: shift function numbers by -1 for F>=3 (after prefix rename).
    shift_targets = []
    for path in renamed:
        if not is_min_ev_file(path, new_prefix):
            continue
        func_num = parse_func_number(path.name, new_prefix)
        if func_num >= 3:
            shift_targets.append((path, func_num))

    # Use temporary names to avoid collisions during shifting.
    shift_temp = {}
    for path, func_num in shift_targets:
        temp = path.with_name(path.name + ".tmp_shift")
        shift_temp[path] = (temp, func_num)

    for src, (tmp, _) in shift_temp.items():
        if args.dry_run:
            print(f"{src.name} -> {tmp.name}")
        else:
            src.rename(tmp)

    for original, (tmp, func_num) in shift_temp.items():
        new_func = func_num - 1
        new_name = f"{new_prefix}_F{new_func}_Min_EV.txt"
        dst = tmp.with_name(new_name)
        if args.dry_run:
            print(f"{tmp.name} -> {dst.name}")
        else:
            tmp.rename(dst)


if __name__ == "__main__":
    main()
