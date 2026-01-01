# app.py
from __future__ import annotations

import json
import os
import hashlib
import io
from typing import Any, Dict

import pandas as pd
import streamlit as st

import storage
from trainer import basic_profile, train_and_evaluate, predict_from_model


st.set_page_config(page_title="ML Management System (POC)", layout="wide")
storage.init_db()


def _load_df(file_path: str) -> pd.DataFrame:
    abs_fp = storage.abs_path(file_path)
    if not os.path.exists(abs_fp):
        raise FileNotFoundError(f"CSV missing on disk:\n{abs_fp}\n\nTip: On Streamlit Cloud, upload the dataset again inside the app.")
    return pd.read_csv(abs_fp)


def _project_selector(projects):
    if not projects:
        return None
    labels = [f'#{p["id"]} — {p["name"]}' for p in projects]
    idx = st.sidebar.selectbox("Select Project", range(len(projects)), format_func=lambda i: labels[i])
    return projects[idx]


# ---------------------------
# Sidebar: Project controls
# ---------------------------
st.sidebar.title("MLMS • POC")
projects = storage.list_projects()

with st.sidebar.expander("➕ Create Project", expanded=(len(projects) == 0)):
    name = st.text_input("Project name", placeholder="Churn Prediction")
    desc = st.text_area("Description", placeholder="Predict customer churn using tabular data.")
    if st.button("Create Project", use_container_width=True):
        if not name.strip():
            st.error("Project name is required.")
        else:
            pid = storage.create_project(name=name, description=desc)
            st.success(f"Created project #{pid}")
            st.rerun()

project = _project_selector(projects)
if not project:
    st.info("Create a project to get started.")
    st.stop()

st.title(f'📁 Project: {project["name"]}')
st.caption(project.get("description", ""))


# ---------------------------
# Session state init (for upload safety)
# ---------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_v1"
if "last_saved_file_hash" not in st.session_state:
    st.session_state.last_saved_file_hash = None


# ---------------------------
# Main Tabs
# ---------------------------
tab_overview, tab_data, tab_experiments, tab_runs, tab_deploy = st.tabs(
    ["Overview", "Data", "Experiments", "Runs", "Deployments"]
)

# ---------------------------
# Overview
# ---------------------------
with tab_overview:
    datasets = storage.list_datasets(project["id"])
    exps = storage.list_experiments(project["id"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Datasets", len(datasets))
    col2.metric("Experiments", len(exps))

    best = None
    best_score = None

    for e in exps:
        runs = storage.list_runs(e["id"])
        for r in runs:
            m = r.get("metrics") or {}
            for key in ["roc_auc", "accuracy", "f1_macro", "r2", "rmse", "mae", "silhouette"]:
                if key in m and isinstance(m[key], (int, float)):
                    score = float(m[key])
                    normalized = -score if key in ("rmse", "mae") else score
                    if best_score is None or normalized > best_score:
                        best_score = normalized
                        best = (e, r, key, score)

    col3.metric("Best Run Found", "Yes" if best else "No")

    st.divider()
    if best:
        e, r, key, score = best
        st.subheader("🏆 Best Run")
        st.write(
            {
                "experiment": f'#{e["id"]} {e["name"]}',
                "run": f'#{r["id"]}',
                "algorithm": r["algorithm"],
                "metric": key,
                "value": score,
                "status": r["status"],
            }
        )
    else:
        st.info("No successful runs yet. Create an experiment and run training.")

# ---------------------------
# Data
# ---------------------------
with tab_data:
    st.subheader("📦 Datasets")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Upload CSV (Preview → Save)")
        dataset_name = st.text_input(
            "Dataset name",
            value="dataset",
            help="Used for versioning (dataset_v1, v2, …)",
        )

        uploaded = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            key=st.session_state.uploader_key,
        )

        if uploaded is None:
            st.info("Upload a CSV to preview it. Then click **Save as new dataset version**.")
        else:
            try:
                file_bytes = uploaded.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()

                df_preview = pd.read_csv(io.BytesIO(file_bytes))
                st.markdown("#### Preview (first 30 rows)")
                st.dataframe(df_preview.head(30), use_container_width=True)

                st.markdown("#### Quick profile")
                summary = basic_profile(df_preview)
                st.json(summary)

                if st.button("✅ Save as new dataset version", type="primary", use_container_width=True):
                    if st.session_state.last_saved_file_hash == file_hash:
                        st.warning("This exact file was already saved. Upload a different file to create a new version.")
                    else:
                        file_path, version = storage.save_dataset_csv(project["id"], dataset_name, file_bytes)
                        dataset_id = storage.create_dataset_record(
                            project_id=project["id"],
                            name=dataset_name,
                            version=version,
                            file_path=file_path,  # stored relative
                            summary=summary,
                        )

                        st.session_state.last_saved_file_hash = file_hash
                        st.session_state.uploader_key = st.session_state.uploader_key + "_reset"

                        st.success(f"Saved dataset #{dataset_id} — {dataset_name} v{version}")
                        st.rerun()

            except Exception as e:
                st.error(f"Upload/preview failed: {e}")

    with right:
        datasets = storage.list_datasets(project["id"])
        if not datasets:
            st.info("No datasets yet. Upload a CSV to begin.")
        else:
            st.markdown("### Dataset Versions")

            df_list = pd.DataFrame(
                [
                    {
                        "dataset_id": d["id"],
                        "name": d["name"],
                        "version": d["version"],
                        "file_exists": os.path.exists(storage.abs_path(d["file_path"])),
                        "rows": (d.get("summary") or {}).get("rows"),
                        "cols": (d.get("summary") or {}).get("cols"),
                        "created_at": d["created_at"],
                    }
                    for d in datasets
                ]
            )
            st.dataframe(df_list, use_container_width=True, hide_index=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("🧹 Remove dataset records with missing files", use_container_width=True):
                    removed = 0
                    for d in datasets:
                        if not os.path.exists(storage.abs_path(d["file_path"])):
                            storage.delete_dataset(d["id"])
                            removed += 1
                    st.success(f"Removed {removed} broken dataset record(s).")
                    st.rerun()

            selected = st.selectbox(
                "Preview dataset",
                options=datasets,
                format_func=lambda d: f'#{d["id"]} — {d["name"]} v{d["version"]}',
            )

            try:
                df_prev = _load_df(selected["file_path"])
            except FileNotFoundError as e:
                st.error(str(e))
                if st.button("🗑️ Delete this broken dataset record", use_container_width=True):
                    storage.delete_dataset(selected["id"])
                    st.success("Deleted dataset record.")
                    st.rerun()
                st.stop()

            st.markdown("### Preview")
            st.dataframe(df_prev.head(50), use_container_width=True)

            st.markdown("### Quick profile")
            st.json(selected.get("summary", {}))

# ---------------------------
# Experiments
# ---------------------------
with tab_experiments:
    st.subheader("🧪 Experiments")

    datasets = storage.list_datasets(project["id"])
    if not datasets:
        st.warning("Upload a dataset first.")
    else:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("### Create Experiment")
            exp_name = st.text_input("Experiment name", value="Baseline 1")

            ds = st.selectbox(
                "Dataset version",
                options=datasets,
                format_func=lambda d: f'#{d["id"]} — {d["name"]} v{d["version"]}',
            )

            # Guard missing file
            if not os.path.exists(storage.abs_path(ds["file_path"])):
                st.error(
                    "Selected dataset file is missing on disk.\n\n"
                    "Fix: Go to Data tab → remove broken records OR upload the dataset again."
                )
                st.stop()

            df = _load_df(ds["file_path"])
            cols = df.columns.tolist()

            problem_type = st.selectbox("Problem type", ["classification", "regression", "clustering"])

            target_col = ""
            if problem_type != "clustering":
                target_col = st.selectbox("Target column", cols)

            metric_options = {
                "classification": ["roc_auc", "accuracy", "f1_macro"],
                "regression": ["rmse", "mae", "r2"],
                "clustering": ["silhouette"],
            }
            metric = st.selectbox("Primary metric", metric_options[problem_type])

            algo_options = {
                "classification": ["logistic_regression", "random_forest", "gradient_boosting"],
                "regression": ["linear_regression", "random_forest", "gradient_boosting"],
                "clustering": ["kmeans"],
            }
            algorithms = st.multiselect(
                "Algorithms",
                algo_options[problem_type],
                default=[algo_options[problem_type][0]],
            )

            config = {"train_test_split": 0.2, "random_state": 42}

            if st.button("Create Experiment", use_container_width=True):
                if not exp_name.strip():
                    st.error("Experiment name is required.")
                elif not algorithms:
                    st.error("Select at least one algorithm.")
                else:
                    eid = storage.create_experiment(
                        project_id=project["id"],
                        name=exp_name,
                        dataset_id=ds["id"],
                        problem_type=problem_type,
                        target_col=target_col,
                        metric=metric,
                        algorithms=algorithms,
                        config=config,
                    )
                    st.success(f"Created experiment #{eid}")
                    st.rerun()

        with right:
            st.markdown("### Existing Experiments")
            exps = storage.list_experiments(project["id"])
            if not exps:
                st.info("No experiments yet.")
            else:
                exp_table = pd.DataFrame(
                    [
                        {
                            "experiment_id": e["id"],
                            "name": e["name"],
                            "dataset_id": e["dataset_id"],
                            "problem_type": e["problem_type"],
                            "target": e["target_col"],
                            "metric": e["metric"],
                            "algorithms": ", ".join(e["algorithms"]),
                            "created_at": e["created_at"],
                        }
                        for e in exps
                    ]
                )
                st.dataframe(exp_table, use_container_width=True, hide_index=True)

# ---------------------------
# Runs
# ---------------------------
with tab_runs:
    st.subheader("🏃 Runs")

    exps = storage.list_experiments(project["id"])
    if not exps:
        st.warning("Create an experiment first.")
    else:
        selected_exp = st.selectbox(
            "Select Experiment",
            options=exps,
            format_func=lambda e: f'#{e["id"]} — {e["name"]} ({e["problem_type"]})',
        )

        exp = storage.get_experiment(selected_exp["id"])
        assert exp is not None

        ds = storage.get_dataset(exp["dataset_id"])
        assert ds is not None

        if not os.path.exists(storage.abs_path(ds["file_path"])):
            st.error(
                "Dataset file missing for this experiment.\n\n"
                "Fix: Go to Data tab → remove broken records OR upload dataset again and create a new experiment."
            )
            st.stop()

        df = _load_df(ds["file_path"])

        st.markdown("### Start a new run")
        algo = st.selectbox("Algorithm", exp["algorithms"])

        default_params: Dict[str, Any] = {}
        if algo in ("random_forest",):
            default_params = {"n_estimators": 300}
        if algo in ("kmeans",):
            default_params = {"n_clusters": 3}

        params_text = st.text_area("Params (JSON)", value=json.dumps(default_params, indent=2), height=120)

        if st.button("▶ Run Training", type="primary"):
            try:
                params = json.loads(params_text) if params_text.strip() else {}
            except Exception:
                st.error("Params must be valid JSON.")
                st.stop()

            run_id = storage.create_run(exp["id"], algo, params)
            storage.update_run(run_id, status="RUNNING", logs_text="Run started...\n")

            with st.spinner("Training in progress..."):
                try:
                    model_dir = os.path.join(storage.ARTIFACTS_DIR, f"run_{run_id}")
                    os.makedirs(model_dir, exist_ok=True)

                    model_path = os.path.join(model_dir, "model.joblib")

                    metrics, logs = train_and_evaluate(
                        df=df,
                        problem_type=exp["problem_type"],
                        target_col=exp["target_col"],
                        metric=exp["metric"],
                        algorithm=algo,
                        model_out_path=model_path,
                        params=params,
                    )

                    # store relative model path in DB
                    storage.update_run(
                        run_id,
                        status="SUCCEEDED",
                        metrics=metrics,
                        logs_text=logs,
                        model_path=model_path,
                    )
                    st.success(f"Run #{run_id} succeeded.")
                except Exception as e:
                    storage.update_run(run_id, status="FAILED", logs_text=f"FAILED: {e}")
                    st.error(f"Run failed: {e}")

            st.rerun()

        st.divider()
        st.markdown("### Run history")

        runs = storage.list_runs(exp["id"])
        if not runs:
            st.info("No runs yet.")
        else:
            runs_table = pd.DataFrame(
                [
                    {
                        "run_id": r["id"],
                        "algorithm": r["algorithm"],
                        "status": r["status"],
                        "metrics": json.dumps(r.get("metrics") or {}),
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                    }
                    for r in runs
                ]
            )
            st.dataframe(runs_table, use_container_width=True, hide_index=True)

            picked_run = st.selectbox(
                "View run detail",
                options=runs,
                format_func=lambda r: f'#{r["id"]} — {r["algorithm"]} [{r["status"]}]',
            )

            st.markdown("### Run details")
            c1, c2 = st.columns([1, 1])
            c1.json(picked_run.get("metrics") or {})
            c2.write({"model_path": picked_run.get("model_path", ""), "params": picked_run.get("params") or {}})

            st.markdown("### Logs")
            st.code(picked_run.get("logs_text") or "", language="text")

# ---------------------------
# Deployments
# ---------------------------
with tab_deploy:
    st.subheader("🚀 Deployments (POC)")

    exps = storage.list_experiments(project["id"])
    if not exps:
        st.info("Create an experiment and run a model first.")
    else:
        all_success_runs = []
        for e in exps:
            for r in storage.list_runs(e["id"]):
                if r["status"] == "SUCCEEDED" and r.get("model_path"):
                    all_success_runs.append((e, r))

        if not all_success_runs:
            st.warning("No successful runs found to deploy.")
        else:
            st.markdown("### Create Deployment from a run")
            choice = st.selectbox(
                "Pick a run",
                options=all_success_runs,
                format_func=lambda t: f'Run #{t[1]["id"]} — {t[1]["algorithm"]} (Exp #{t[0]["id"]}: {t[0]["name"]})',
            )
            exp, run = choice

            dep_name = st.text_input("Deployment name", value=f"{project['name']} • {run['algorithm']} • v1")

            if st.button("Create Deployment", use_container_width=True):
                dep_id = storage.create_deployment(project_id=project["id"], name=dep_name, run_id=run["id"])
                st.success(f"Deployment #{dep_id} created (ACTIVE).")
                st.rerun()

            st.divider()
            st.markdown("### Existing Deployments")
            deps = storage.list_deployments(project["id"])
            if deps:
                dep_df = pd.DataFrame(deps)
                st.dataframe(dep_df, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Test a deployed model (in-app)")
            deps = storage.list_deployments(project["id"])
            if deps:
                _toggle = st.selectbox(
                    "Select deployment",
                    options=deps,
                    format_func=lambda d: f'#{d["id"]} — {d["name"]} [{d["status"]}]',
                )
                dep = _toggle
                run_obj = storage.get_run(dep["run_id"])
                if not run_obj or not run_obj.get("model_path"):
                    st.error("Run/model missing for this deployment.")
                else:
                    abs_model_path = storage.abs_path(run_obj["model_path"])
                    if not os.path.exists(abs_model_path):
                        st.error(
                            f"Model file missing on disk:\n{abs_model_path}\n\n"
                            "If you're on Streamlit Cloud, you must train the model inside the deployed app."
                        )
                        st.stop()

                    input_json = st.text_area(
                        "Input features JSON (one row)",
                        value='{"feature1": 1, "feature2": "A"}',
                        height=120,
                    )
                    if st.button("Send Test Request", type="primary"):
                        try:
                            row = json.loads(input_json)
                            out = predict_from_model(abs_model_path, row)
                            st.json(out)
                            st.code(
                                "curl (placeholder):\n"
                                "curl -X POST <endpoint> -H 'Content-Type: application/json' -d '<payload>'",
                                language="bash",
                            )
                        except Exception as e:
                            st.error(f"Test failed: {e}")
            else:
                st.info("No deployments yet.")
