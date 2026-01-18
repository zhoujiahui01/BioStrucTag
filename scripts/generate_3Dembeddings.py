from pathlib import Path
import numpy as np
import torch
from pyuul import VolumeMaker, utils
from tqdm import tqdm


def tensor_to_numpy(tensor):
    if hasattr(tensor, "is_sparse") and tensor.is_sparse:
        tensor = tensor.to_dense()
    return tensor.cpu().numpy()


def process_pdbs(pdb_files, output_dir, voxels, device, desc):
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdb in tqdm(pdb_files, desc=desc, ncols=80):
        try:
            coords, atname = utils.parsePDB(str(pdb))
            ch = utils.atomlistToChannels(atname)
            rad = utils.atomlistToRadius(atname)

            coords = coords.to(device)
            ch = ch.to(device)
            rad = rad.to(device)

            voxel = voxels(coords, rad, ch)

            out_file = output_dir / f"{pdb.stem}_voxel.npy"
            np.save(out_file, tensor_to_numpy(voxel))

        except Exception as e:
            print(f"[FAILED] {pdb.name}: {e}")


def main():
    ROOT = Path(__file__).resolve().parents[1]

    INPUT = ROOT / "data" / "activesite"
    OUTPUT = ROOT / "outputs" / "3D_embeddings"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Voxels = VolumeMaker.Voxels(device=device, sparse=True)

    # -------------------------------------------------
    # 1) Process PDB files directly under activesite/
    # -------------------------------------------------
    root_pdbs = list(INPUT.glob("*.pdb"))

    if root_pdbs:
        print(f"[INFO] Found {len(root_pdbs)} PDB files in {INPUT}")
        process_pdbs(
            pdb_files=root_pdbs,
            output_dir=OUTPUT,
            voxels=Voxels,
            device=device,
            desc="Processing activesite/"
        )

    # -------------------------------------------------
    # 2) Process subfolders (if any)
    # -------------------------------------------------
    subdirs = [d for d in INPUT.iterdir() if d.is_dir()]

    for subdir in subdirs:
        pdbs = list(subdir.glob("*.pdb"))
        if not pdbs:
            continue

        print(f"[INFO] Found {len(pdbs)} PDB files in {subdir.name}/")

        process_pdbs(
            pdb_files=pdbs,
            output_dir=OUTPUT / subdir.name,
            voxels=Voxels,
            device=device,
            desc=f"Processing {subdir.name}/"
        )

    if not root_pdbs and not subdirs:
        print(f"[WARNING] No PDB files found in {INPUT}")

    print("\n3D embeddings generation completed.")


if __name__ == "__main__":
    main()
