"""Binary IDS record-by-record replay dashboard."""
from pathlib import Path
import os
import time
import joblib
import numpy as np
import pandas as pd
import psutil
import streamlit as st
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "UNSW_NB15_testing-set.csv"
MODEL = ROOT / "models" / "best_ids_model.pkl"
SCALER = ROOT / "models" / "scaler.pkl"
FEATURES = ROOT / "models" / "selected_feature_names.pkl"

st.set_page_config(page_title="Real-Time IDS Replay", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.stApp {background:#fff;color:#172033;} [data-testid="stHeader"] {background:rgba(255,255,255,.96);}
.hero {padding:1.3rem 1.6rem;border:1px solid #dbeafe;border-radius:18px;background:linear-gradient(110deg,#eff6ff,#fff);margin-bottom:1rem;}
.hero h1 {color:#123b66;margin:0;} .hero p {color:#536579;margin:.35rem 0 0;}
[data-testid="stMetric"] {background:#f8fbff;border:1px solid #e2eaf3;border-radius:12px;padding:.6rem;}
</style><div class="hero"><h1>🛡️ Real-Time Intrusion Detection Demo</h1><p>Replay connected UNSW-NB15 test traffic one record at a time.</p></div>""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL), joblib.load(SCALER), joblib.load(FEATURES)

@st.cache_data
def load_data():
    return pd.read_csv(DATA).dropna(subset=["label"]).drop_duplicates().reset_index(drop=True)

def encode_full_input(raw, all_feature_names):
    frame = raw.drop(columns=["id", "attack_cat", "label"], errors="ignore")
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.get_dummies(frame, dtype=float).reindex(columns=all_feature_names, fill_value=0)

try:
    model, scaler, names = load_artifacts()
    test_data = load_data()
except FileNotFoundError:
    st.error("Model files are missing. Run notebook 02 first to create the model.")
    st.stop()

st.sidebar.title("Replay controls")
st.sidebar.caption("The testing set is already connected — no upload is needed.")
start, end = st.sidebar.slider("Select record range", 0, len(test_data) - 1, (0, min(199, len(test_data) - 1)))
delay = st.sidebar.slider("Playback delay per record", 0.0, 2.0, 0.25, 0.05)
st.sidebar.write(f"**Records selected:** {end - start + 1:,}")
run = st.sidebar.button("▶ Start real-time replay", type="primary", use_container_width=True)

if not run:
    st.info("Choose a range and speed, then select **Start real-time replay**.")
    st.stop()

# The scaler expects all encoded inputs. Scale those first, then keep the 30
# selected inputs expected by the saved model.
full_features = encode_full_input(test_data.iloc[start:end + 1], list(scaler.feature_names_in_))
selected_indices = [full_features.columns.get_loc(name) for name in names]
features = scaler.transform(full_features)[:, selected_indices]
total = len(features)
progress = st.progress(0, text="Preparing replay…")
c1, c2, c3, c4 = st.columns(4)
processed_box, attacks_box, normal_box, accuracy_box = c1.empty(), c2.empty(), c3.empty(), c4.empty()

st.subheader("Live System & Hardware Resource Usage")
r1, r2, r3, r4 = st.columns(4)
ram_box, cpu_box, latency_box, throughput_box = r1.empty(), r2.empty(), r3.empty(), r4.empty()

current = st.empty()
st.subheader("Recent detections")
recent = st.empty()

process = psutil.Process(os.getpid())
records, attacks, correct = [], 0, 0
total_infer_time, peak_ram = 0.0, 0.0

for i in range(total):
    row = start + i
    t0 = time.perf_counter()
    prediction = int(model.predict(features[i:i + 1])[0])
    dt = time.perf_counter() - t0
    total_infer_time += dt

    actual = int(test_data.iloc[row]["label"])
    predicted_label = "Attack" if prediction else "Normal"
    actual_label = "Attack" if actual else "Normal"
    attacks += prediction
    correct += prediction == actual
    completed = i + 1

    current_ram = process.memory_info().rss / (1024 * 1024)
    peak_ram = max(peak_ram, current_ram)
    cpu_pct = process.cpu_percent(interval=None)

    if prediction:
        current.error(f"🚨 Record {row:,}: **Attack detected**")
    else:
        current.success(f"✅ Record {row:,}: **Normal traffic**")

    processed_box.metric("Processed", f"{completed:,} / {total:,}")
    attacks_box.metric("Attacks flagged", f"{attacks:,}")
    normal_box.metric("Normal traffic", f"{completed - attacks:,}")
    accuracy_box.metric("Running accuracy", f"{correct / completed:.1%}")

    ram_box.metric("RAM (RSS)", f"{current_ram:.1f} MB")
    cpu_box.metric("CPU Usage", f"{cpu_pct:.1f}%")
    latency_box.metric("Packet Latency", f"{dt * 1000:.3f} ms")
    throughput_box.metric("Avg Latency", f"{(total_infer_time / completed) * 1000:.3f} ms")

    records.append({"Record": row, "Prediction": predicted_label, "True label": actual_label,
                    "Correct": "Yes" if prediction == actual else "No",
                    "Latency (ms)": round(dt * 1000, 3)})
    recent.dataframe(pd.DataFrame(records[-15:][::-1]), hide_index=True, use_container_width=True)
    progress.progress(completed / total, text=f"Analysing record {completed:,} of {total:,}")
    if delay:
        time.sleep(delay)

results = pd.DataFrame(records)
y_true = (results["True label"] == "Attack").astype(int)
y_pred = (results["Prediction"] == "Attack").astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
fpr = fp / (fp + tn) if fp + tn else None
avg_latency_ms = (total_infer_time / total) * 1000 if total > 0 else 0
throughput_fps = total / total_infer_time if total_infer_time > 0 else 0

st.success(f"Replay complete — {total:,} records analysed.")
st.divider()
st.subheader("Detection & Replay Analysis")
a, b, c, d = st.columns(4)
a.metric("Accuracy", f"{(tp + tn) / total:.2%}")
b.metric("True attacks found", f"{tp:,}")
c.metric("False alarms", f"{fp:,}")
d.metric("False-positive rate", f"{fpr:.2%}" if fpr is not None else "N/A")
st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual Normal", "Actual Attack"], columns=["Predicted Normal", "Predicted Attack"]), use_container_width=True)
if fpr is None:
    st.caption("False-positive rate is N/A because this range has no actual Normal records.")

st.subheader("System Performance & Hardware Resource Summary")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Avg Inference Latency", f"{avg_latency_ms:.4f} ms / packet")
h2.metric("Inference Throughput", f"{throughput_fps:,.0f} pkts / sec")
h3.metric("Peak RAM Usage", f"{peak_ram:.1f} MB")
h4.metric("Total Model Infer Time", f"{total_infer_time:.4f} s")

st.download_button("Download complete prediction results (CSV)", results.to_csv(index=False).encode(), f"ids_replay_{start}_{end}.csv", "text/csv")
