import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm


def main():
    ROOT = Path(__file__).resolve().parents[1]
    OUTPUTS = ROOT / "outputs"
    MODELS = ROOT / "models"

    parser = argparse.ArgumentParser(
        description="Generate per-sequence 1D ESM embeddings"
    )
    parser.add_argument(
        "--input",
        default="sequences.csv",
        help="CSV file under outputs/ containing Sequences and Mutations"
    )
    parser.add_argument(
        "--outdir",
        default="1D_embeddings",
        help="Output directory under outputs/"
    )
    args = parser.parse_args()

    input_csv = OUTPUTS / args.input
    outdir = OUTPUTS / args.outdir
    cache_dir = MODELS / "esm"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if "Sequences" not in df.columns or "Mutations" not in df.columns:
        raise ValueError("CSV must contain 'Sequences' and 'Mutations' columns")

    # Normalize mutation names for filenames
    df["Mutations"] = (
        df["Mutations"]
        .fillna("WT")
        .str.replace("/", "_", regex=False)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "facebook/esm2_t12_35M_UR50D"
    model = EsmModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        add_pooling_layer=False
    ).to(device)
    model.eval()

    tokenizer = EsmTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ESM encoding", ncols=80):
        mutation = row["Mutations"]
        sequence = row["Sequences"]

        out_file = outdir / f"{mutation}.pt"
        if out_file.exists():
            continue  # skip existing embeddings

        try:
            inputs = tokenizer(
                sequence,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

            torch.save(
                {
                    "embedding": embedding,
                    "mutation": mutation,
                    "sequence": sequence
                },
                out_file
            )

        except Exception as e:
            print(f"[FAILED] {mutation}: {e}")

    print(f"\n1D embeddings saved to: {outdir}")


if __name__ == "__main__":
    main()
