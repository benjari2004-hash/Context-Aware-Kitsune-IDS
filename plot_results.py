import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_FILE = Path(__file__).with_name("results.csv")
PLOT_FILE = Path(__file__).with_name("results_plot.png")


def load_results():
    ids = []
    scores = []
    thresholds = []
    labels = []

    with open(RESULTS_FILE, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(line for line in csv_file if not line.startswith('#'))
        for row in reader:
            packet_id = int(row["packet_id"])
            score = float(row["score"])
            threshold = float(row["threshold"])
            label = row["label"]

            ids.append(packet_id)
            scores.append(score)
            thresholds.append(threshold)
            labels.append(label)

    return ids, scores, thresholds, labels


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Run main_ids.py first to create {RESULTS_FILE}")

    ids, scores, thresholds, labels = load_results()
    print("Total rows:", len(ids))

    anomaly_x = [packet_id for packet_id, label in zip(ids, labels) if label == "ANOMALY"]
    anomaly_y = [score for score, label in zip(scores, labels) if label == "ANOMALY"]

    plt.figure(figsize=(14, 7))
    plt.plot(ids, scores, label="Score", color="steelblue", linewidth=1, alpha=0.7)
    plt.plot(ids, thresholds, label="Threshold", color="darkorange", linewidth=1)
    plt.scatter(anomaly_x, anomaly_y, color="red", label="Anomaly", s=10, zorder=3)

    plt.yscale("symlog")
    plt.xlabel("Packet ID")
    plt.ylabel("Score / Threshold")
    plt.title("Kitsune IDS Anomaly Detection Results")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=300)

    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
