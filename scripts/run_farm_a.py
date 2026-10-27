"""Train the EnergyFaultDetector on every Wind Farm A event and save the anomaly scores.

For each event: train on the training part, predict on the test part, save
  - scores/event_<id>_anomaly_scores.csv  (event_id, time_stamp, anomaly_score, threshold, predicted_anomaly)
  - models/event_<id>/                    (the trained model)
At the end: metrics.csv with precision, recall and F1 per event, plus one row for all events together.

Run from the project folder:
    python scripts/run_farm_a.py --data data/CARE_To_Compare --config configs/config.yaml --out results/farm_a_baseline
    (add --events 40 3 to run only some events)

    sbatch scripts/job.slurm python scripts/run_farm_a.py --data data/CARE_To_Compare --config configs/windfarm_A.yaml --out results/26102026_161432
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from energy_fault_detector import Config, FaultDetector
from energy_fault_detector.evaluation import Care2CompareDataset

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--events", type=int, nargs="*")
args = ap.parse_args()

out = Path(args.out)
(out / "scores").mkdir(parents=True, exist_ok=True) # create farm_A scores folder under /results

c2c = Care2CompareDataset(path=args.data)
events = c2c.event_info_all[c2c.event_info_all["wind_farm"] == "A"]
if args.events:
    events = events[events["event_id"].isin(args.events)] # if some specific events are specified, if not run all

all_scores = []
for i, (_, event_row) in enumerate(events.iterrows(), 1):
    event_id = int(event_row["event_id"])
    print(f"[{i}/{len(events)}] event {event_id} ({event_row['event_label']})", flush=True)

    # 1. load data
    x_train, train_status, x_test, test_status = c2c.load_and_format_event_dataset(event_id=event_id, index_column="time_stamp")

    # 2. train
    config = Config(args.config)
    c2c.update_c2c_config(config=config, wind_farm="A")
    model = FaultDetector(config=config, model_directory=str(out / "models"))
    model.fit(sensor_data=x_train, normal_index=train_status, save_models=False)
    model.save_models(model_name=f"event_{event_id}", overwrite=True)

    # 3. predict
    predictions = model.predict(sensor_data=x_test)
    threshold = np.asarray(model.threshold_selector.threshold).reshape(-1)
    if threshold.size == 1:
        threshold = np.repeat(threshold, len(predictions.anomaly_score))

    scores = pd.DataFrame({
        "event_id": event_id,
        "time_stamp": predictions.anomaly_score.index,
        "anomaly_score": predictions.anomaly_score.values,
        "threshold": threshold,
        "predicted_anomaly": predictions.predicted_anomalies.values,
    })

    # true label: inside the fault window of an anomaly event (Farm A: status codes are ignored)
    in_window = (scores["time_stamp"] >= event_row["event_start"]) & (scores["time_stamp"] <= event_row["event_end"])
    scores["true_anomaly"] = in_window & (event_row["event_label"] == "anomaly")

    scores.to_csv(out / "scores" / f"event_{event_id}_anomaly_scores.csv", index=False)
    all_scores.append(scores)

# 4. basic metrics
def metrics(df, name, label):
    y_true, y_pred = df["true_anomaly"].astype(bool), df["predicted_anomaly"].astype(bool)
    return {
        "event_id": name,
        "label": label,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "share_flagged": y_pred.mean(),   # for normal events: share of false alarms
    }

combined = pd.concat(all_scores, ignore_index=True)
combined.to_csv(out / "all_events_anomaly_scores.csv", index=False)

labels = events.set_index("event_id")["event_label"]
rows = [metrics(s, int(s["event_id"].iloc[0]), labels[int(s["event_id"].iloc[0])]) for s in all_scores]
rows.append(metrics(combined, "all", "all"))
table = pd.DataFrame(rows).round(3)
# precision / recall / F1 are not meaningful for normal events (they contain no true anomalies)
table.loc[table["label"] == "normal", ["precision", "recall", "f1"]] = np.nan
table.to_csv(out / "metrics.csv", index=False)
print(table.to_string(index=False))