import csv
import sys
import argparse
from pathlib import Path

VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

class MutationError(Exception):
    pass


def read_fasta(fasta_path):
    seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                seq.append(line.strip())
    sequence = ''.join(seq).upper()
    if not sequence:
        raise MutationError("FASTA file contains no sequence")
    return sequence


def validate_mutation_format(mutation):
    if len(mutation) < 3:
        raise MutationError(f"Mutation too short: {mutation}")
    if not mutation[1:-1].isdigit():
        raise MutationError(f"Position must be numeric: {mutation}")
    if mutation[0] not in VALID_AMINO_ACIDS or mutation[-1] not in VALID_AMINO_ACIDS:
        raise MutationError(f"Invalid amino acid in {mutation}")


def process_mutations(input_csv, wt_sequence, output_csv):
    processed = []
    success, errors = 0, 0

    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if 'Mutations' not in reader.fieldnames:
            raise MutationError("Missing 'Mutations' column")

        for i, row in enumerate(reader, 1):
            muts = row['Mutations'].strip().upper()
            try:
                if muts == "WT":
                    processed.append({
                        "Mutations": "WT",
                        "Sequences": wt_sequence,
                        "Status": "Wildtype"
                    })
                    success += 1
                    continue

                seq = wt_sequence
                applied = []

                for m in muts.split('/'):
                    validate_mutation_format(m)
                    o, pos, n = m[0], int(m[1:-1]), m[-1]
                    if seq[pos - 1] != o:
                        raise MutationError(f"Mismatch at {pos}")
                    seq = seq[:pos - 1] + n + seq[pos:]
                    applied.append(m)

                processed.append({
                    "Mutations": "/".join(applied),
                    "Sequences": seq,
                    "Status": "Success"
                })
                success += 1

            except Exception as e:
                errors += 1
                processed.append({
                    "Mutations": muts,
                    "Sequences": "INVALID",
                    "Status": str(e)
                })
                print(f"[Row {i}] {e}", file=sys.stderr)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Mutations", "Sequences", "Status"])
        writer.writeheader()
        writer.writerows(processed)

    print(f"Done: {success} success, {errors} errors")
    print(f"Saved to {output_csv}")


def main():
    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    OUTPUTS = ROOT / "outputs"
    OUTPUTS.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="mutations.csv")
    parser.add_argument("--fasta", default="wildtype.fasta")
    parser.add_argument("--output", default="sequences.csv")
    args = parser.parse_args()

    input_csv = DATA / args.input
    fasta_file = DATA / args.fasta
    output_csv = OUTPUTS / args.output

    wt_seq = read_fasta(fasta_file)
    process_mutations(input_csv, wt_seq, output_csv)


if __name__ == "__main__":
    main()
