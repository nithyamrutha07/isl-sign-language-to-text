import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

input_file = BASE_DIR / "dataset" / "alphabet_data_v2.csv"
output_file = BASE_DIR / "dataset" / "alphabet_data_cleaned.csv"

# Letters that use only one hand
single_hand_letters = {"C", "L", "I", "U", "V", "O"}

kept = 0
removed = 0

with open(input_file, "r", newline="") as infile:

    reader = csv.DictReader(infile)

    with open(output_file, "w", newline="") as outfile:

        writer = csv.DictWriter(
            outfile,
            fieldnames=reader.fieldnames
        )

        writer.writeheader()

        for row in reader:

            left = row["left_present"] == "1"
            right = row["right_present"] == "1"

            if row["label"] in single_hand_letters:
                valid = left != right
            else:
                valid = left and right

            if valid:
                writer.writerow(row)
                kept += 1
            else:
                removed += 1

print("Dataset cleaning completed!")
print("Samples kept:", kept)
print("Samples removed:", removed)
print("Cleaned file:", output_file)