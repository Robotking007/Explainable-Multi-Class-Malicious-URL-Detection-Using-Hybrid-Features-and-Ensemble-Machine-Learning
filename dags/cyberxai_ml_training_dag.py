"""
Master Apache Airflow DAG: CyberXAI ML Training & Deployment Pipeline.

Pipeline Architecture:
  1. validate_data_task
     └──> 2. extract_features_task
          └──> 3. train_model_task
               └──> 4. evaluate_quality_gate_task
                    └──> 5. deploy_to_huggingface_task

Schedule: Weekly or On-Demand
Tags: ['cyberxai', 'mlops', 'url-detection', 'pytorch', 'huggingface']
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Import task callables
from dags.tasks.data_validation import validate_dataset
from dags.tasks.feature_engineering import extract_features
from dags.tasks.model_training import train_model
from dags.tasks.model_evaluation import evaluate_model
from dags.tasks.model_registry_sync import sync_to_huggingface

# Safe Airflow import
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None
    PythonOperator = None


default_args = {
    "owner": "cyberxai-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2026, 1, 1),
}


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="cyberxai_url_ml_training_pipeline",
        default_args=default_args,
        description="End-to-end MLOps pipeline for CyberXAI HybridURLNet training, validation, and HF deployment",
        schedule_interval=None,  # Trigger on-demand or configure e.g. '@weekly'
        catchup=False,
        max_active_runs=1,
        tags=["cyberxai", "mlops", "url-detection", "pytorch", "huggingface"],
    ) as dag:

        # Stage 1: Data Validation & Pre-flight Integrity
        t1_validate = PythonOperator(
            task_id="validate_dataset",
            python_callable=validate_dataset,
        )

        # Stage 2: 45 Hybrid Features Extraction & Stratified Sampling
        t2_features = PythonOperator(
            task_id="extract_hybrid_features",
            python_callable=extract_features,
            op_kwargs={"sample_size": 250000},
        )

        # Stage 3: PyTorch Deep Learning Model Training (HybridURLNet)
        t3_train = PythonOperator(
            task_id="train_hybridurlnet_model",
            python_callable=train_model,
            op_kwargs={"epochs": 6, "batch_size": 1024},
        )

        # Stage 4: Evaluation & MLOps Quality Gate Check
        t4_evaluate = PythonOperator(
            task_id="evaluate_and_quality_gate",
            python_callable=evaluate_model,
            op_kwargs={"min_accuracy": 0.82, "min_f1": 0.78},
        )

        # Stage 5: Synchronize Approved Artifacts to Hugging Face Hub
        t5_deploy = PythonOperator(
            task_id="deploy_to_huggingface_hub",
            python_callable=sync_to_huggingface,
            op_kwargs={"repo_id": "siddudavant/cyberxai-models"},
        )

        # Set DAG execution dependencies
        t1_validate >> t2_features >> t3_train >> t4_evaluate >> t5_deploy


def run_pipeline_standalone():
    """Fallback runner allowing direct local execution without an active Airflow daemon."""
    print("=== Running CyberXAI ML Pipeline Standalone ===")
    print("\n--- Stage 1: Validate Dataset ---")
    validate_dataset()

    print("\n--- Stage 2: Extract Hybrid Features ---")
    extract_features(sample_size=10000)

    print("\n--- Stage 3: Train HybridURLNet Model ---")
    train_model(epochs=3, batch_size=512)

    print("\n--- Stage 4: Evaluate & Quality Gate ---")
    evaluate_model(min_accuracy=0.70, min_f1=0.70)

    print("\n--- Stage 5: Deploy to Hugging Face Hub ---")
    sync_to_huggingface()
    print("\n=== Pipeline Execution Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline_standalone()
