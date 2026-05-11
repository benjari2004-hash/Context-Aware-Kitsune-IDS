"""
Injects realistic false-positive patterns into results.csv to create FP pressure
for evaluating the feedback learning system.  The injected file keeps all original
columns and adds a 'true_label' column (NORMAL / ANOMALY) for ground-truth use.
"""
import csv
import io
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MY_IDS = os.path.dirname(_HERE)
if _MY_IDS not in sys.path:
    sys.path.insert(0, _MY_IDS)

try:
    from decision_layer.decision_engine import make_decision as _make_decision
except ImportError:
    def _make_decision(score, risk, attack_type, freq):
        if risk > 2.0:
            return {"severity": 5, "severity_label": "CRITICAL", "action": "BLOCK", "reason": ""}
        if risk > 1.5:
            return {"severity": 4, "severity_label": "HIGH",     "action": "RATE_LIMIT", "reason": ""}
        if risk > 1.0:
            return {"severity": 3, "severity_label": "MEDIUM",   "action": "ALERT_HIGH", "reason": ""}
        return {"severity": 2, "severity_label": "LOW", "action": "ALERT_LOW", "reason": ""}


def _read_csv_skip_comments(path: str):
    """Return (fieldnames, list-of-dicts) from a CSV that may have leading # lines."""
    with open(path, "r", encoding="utf-8") as fh:
        content = "".join(line for line in fh if not line.lstrip().startswith("#"))
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    return reader.fieldnames or [], rows


def _read_ground_truth(path: str) -> dict:
    gt = {}
    if not os.path.exists(path):
        return gt
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    gt[int(parts[0].strip())] = parts[1].strip().upper()
                except ValueError:
                    pass
    return gt


class FPInjector:
    """
    Injects realistic false-positive patterns into results.csv to create
    FP pressure for evaluating the feedback learning system.
    """

    PATTERNS = {
        "backup_burst": {
            "score_range": (0.25, 0.45),
            "risk_range":  (1.2,  2.0),
            "freq": 60,
            "description": "Nightly backup",
        },
        "dns_flood": {
            "score_range": (0.15, 0.35),
            "risk_range":  (0.8,  1.8),
            "freq": 200,
            "description": "DNS resolver",
        },
        "iot_chatter": {
            "score_range": (0.20, 0.40),
            "risk_range":  (1.0,  1.6),
            "freq": 12,
            "description": "IoT heartbeat",
        },
        "dev_scanner": {
            "score_range": (0.30, 0.60),
            "risk_range":  (1.3,  2.2),
            "freq": 25,
            "description": "Auth vuln scan",
        },
        "video_stream": {
            "score_range": (0.18, 0.38),
            "risk_range":  (0.9,  1.7),
            "freq": 80,
            "description": "Video conference",
        },
        "vpn_tunnel": {
            "score_range": (0.22, 0.42),
            "risk_range":  (1.1,  1.9),
            "freq": 15,
            "description": "VPN tunnel",
        },
        "microservice_burst": {
            "score_range": (0.35, 0.55),
            "risk_range":  (1.5,  2.5),
            "freq": 100,
            "description": "K8s pod restart",
        },
    }

    # Minimum risk that guarantees hard_override_risk detection
    _DETECT_MIN_RISK = 1.55

    def inject(
        self,
        results_path: str,
        ground_truth_path: str,
        output_path: str,
        num_fp_packets: int = 2000,
        seed: int = 42,
    ) -> str:
        """
        1. Read results.csv (skipping comment lines).
        2. Add true_label to every row from ground_truth.csv.
        3. Generate num_fp_packets synthetic packets (risk > 1.55 so they
           are always detected as ANOMALY), with true_label = NORMAL.
        4. Randomly insert FP packets within the detection phase.
        5. Write to output_path and return it.
        """
        rng = random.Random(seed)

        fieldnames, rows = _read_csv_skip_comments(results_path)
        gt = _read_ground_truth(ground_truth_path)

        # ── Add true_label to existing rows ──────────────────────────────
        for row in rows:
            lbl = row.get("label", "").upper()
            pid = int(row.get("packet_id", 0))
            if lbl == "ANOMALY":
                row["true_label"] = gt.get(pid, "ANOMALY")
            elif lbl == "NORMAL":
                row["true_label"] = "NORMAL"
            else:
                row["true_label"] = ""

        # ── Compute typical threshold for injected rows ───────────────────
        thresh_vals = [float(r.get("threshold", 0.5)) for r in rows
                       if r.get("label", "").upper() == "ANOMALY"]
        typical_thresh = round(sum(thresh_vals) / len(thresh_vals), 4) if thresh_vals else 0.5

        max_pid = max(
            (int(r.get("packet_id", 0)) for r in rows
             if r.get("label", "").upper() != "TRAIN"),
            default=100000,
        )

        # ── Generate synthetic FP packets ─────────────────────────────────
        pattern_names = list(self.PATTERNS.keys())
        fp_packets = []

        for i in range(num_fp_packets):
            pname = rng.choice(pattern_names)
            pat = self.PATTERNS[pname]

            r_lo = max(pat["risk_range"][0], self._DETECT_MIN_RISK)
            r_hi = max(pat["risk_range"][1], r_lo + 0.05)
            score = round(rng.uniform(*pat["score_range"]), 4)
            risk  = round(rng.uniform(r_lo, r_hi), 4)
            freq  = pat["freq"]

            dec = _make_decision(score, risk, "", freq)
            pid = max_pid + i + 1

            cf_text = f"If risk were 1.40 (↓ from {risk:.4f}), action would change from {dec['action']} to ALLOW"

            fp_packets.append({
                "packet_id":      pid,
                "score":          score,
                "risk":           risk,
                "threshold":      typical_thresh,
                "label":          "ANOMALY",
                "attack_type":    "",
                "attack_reason":  pat["description"],
                "ara_active":     "False",
                "detection_path": "hard_override_risk",
                "severity":       dec["severity"],
                "action":         dec["action"],
                "cf_pass1":       cf_text,
                "cf_pass2":       "",
                "cf_temporal":    "",
                "true_label":     "NORMAL",
            })

        # ── Randomly insert FP packets into the detection phase ───────────
        train_rows  = [r for r in rows if r.get("label", "").upper() == "TRAIN"]
        detect_rows = [r for r in rows if r.get("label", "").upper() != "TRAIN"]

        for fp in fp_packets:
            pos = rng.randint(0, len(detect_rows))
            detect_rows.insert(pos, fp)

        all_rows = train_rows + detect_rows

        # ── Write output ──────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        out_fields = list(fieldnames)
        if "true_label" not in out_fields:
            out_fields.append("true_label")

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"[FPInjector] {num_fp_packets} FP packets injected → {output_path}")
        print(f"             Total rows: {len(all_rows)}")
        return output_path
