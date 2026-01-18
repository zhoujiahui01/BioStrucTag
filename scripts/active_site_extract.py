import argparse
from pathlib import Path
import pymol
from pymol import cmd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Center coordinates of the active site (in Å)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    struct_dir = root / "data" / "structures"
    align_dir = root / "data" / "alignment"
    active_dir = root / "data" / "activesite"

    align_dir.mkdir(parents=True, exist_ok=True)
    active_dir.mkdir(parents=True, exist_ok=True)

    pymol.finish_launching(["pymol", "-cq"])

    pdb_files = sorted(struct_dir.glob("*.pdb"))
    if not pdb_files:
        raise RuntimeError("No pdb files found in data/structures")

    # Load reference
    ref_file = pdb_files[0]
    cmd.load(str(ref_file), "ref")

    # Align all structures
    for pdb in pdb_files:
        name = pdb.stem
        cmd.load(str(pdb), name)
        if name != "ref":
            cmd.align(name, "ref")
        cmd.save(str(align_dir / f"{name}_aligned.pdb"), name)

    # Crop active site
    x0, y0, z0 = args.center
    selection_box = (
        f"(x > {x0 - 10} and x < {x0 + 10}) and "
        f"(y > {y0 - 10} and y < {y0 + 10}) and "
        f"(z > {z0 - 10} and z < {z0 + 10})"
    )

    cmd.delete("all")

    for pdb in align_dir.glob("*_aligned.pdb"):
        name = pdb.stem.replace("_aligned", "")
        cmd.load(str(pdb), name)
        cmd.select("active_site", selection_box)
        cmd.save(str(active_dir / f"{name}_activesite.pdb"), "active_site")
        cmd.delete(name)
        cmd.delete("active_site")

    cmd.quit()


if __name__ == "__main__":
    main()
