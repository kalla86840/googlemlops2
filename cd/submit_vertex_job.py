#!/usr/bin/env python3
import argparse
from google.cloud import aiplatform


def main() -> None:
    p = argparse.ArgumentParser(description="Submit a Vertex AI CustomJob from a local script.")
    p.add_argument("--project", required=True, help="GCP project id, e.g. inner-domain-397315")
    p.add_argument("--region", required=True, help="Vertex region, e.g. us-west2")
    p.add_argument("--data-uri", required=True, help="GCS URI to training data CSV")
    p.add_argument(
        "--staging-bucket",
        required=True,
        help="GCS bucket used by Vertex to stage packaged code, e.g. gs://...-vertex-staging-...",
    )

    # Optional knobs
    p.add_argument("--display-name", default="calls-sales-training", help="Vertex job display name")
    p.add_argument("--machine-type", default="n1-standard-4", help="Machine type")
    p.add_argument("--replica-count", type=int, default=1, help="Number of replicas")

    args, unknown = p.parse_known_args()

    # Initialize Vertex AI
    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.staging_bucket,
    )

    # IMPORTANT:
    # For CustomJob.from_local_script you must use a Vertex-supported "python package training" image.
    # This is the supported sklearn training image (fixes the error you saw).
    container_uri = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest"

    # Path to your training script inside the repo
    script_path = "src/train_calls_sales.py"

    # Build args passed to your training script
    # - We always pass --data-uri plus any extra unknown args the user included.
    train_args = ["--data-uri", args.data_uri] + unknown

    print("Creating CustomJob...")
    job = aiplatform.CustomJob.from_local_script(
        display_name=args.display_name,
        script_path=script_path,
        container_uri=container_uri,
        args=train_args,
        replica_count=args.replica_count,
        machine_type=args.machine_type,
    )

    print("Submitting job...")
    job.run(sync=True)

    print("Vertex job complete:", job.resource_name)


if __name__ == "__main__":
    main()
