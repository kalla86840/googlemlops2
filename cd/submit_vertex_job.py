import argparse
from datetime import datetime
from google.cloud import aiplatform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-uri", required=True)
    parser.add_argument("--staging-bucket", required=True)
    args = parser.parse_args()

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.staging_bucket,
    )

    # Vertex-supported training container (sklearn)
    container_uri = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest"

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    # This MUST be provided to the wrapper task.py
    model_dir = f"{args.staging_bucket}/model-artifacts/{run_id}"

    job = aiplatform.CustomJob.from_local_script(
        display_name=f"calls-sales-training-{run_id}",
        script_path="src/train_calls_sales.py",
        container_uri=container_uri,
        args=[
            "--data-uri", args.data_uri,
            "--model-dir", model_dir,
            "--target-col", "sales",
        ],
        replica_count=1,
        machine_type="n1-standard-4",

        # Ensure extra deps are installed (fixes python-json-logger issue)
        requirements=[
            "pandas",
            "scikit-learn",
            "joblib",
            "gcsfs",
            "python-json-logger",
        ],
    )

    job.run(sync=True)
    print("Vertex job complete:", job.resource_name)


if __name__ == "__main__":
    main()
