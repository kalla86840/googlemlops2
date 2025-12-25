import argparse
from datetime import datetime
from google.cloud import aiplatform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--staging-bucket", required=True)
    parser.add_argument("--data-uri", required=True)
    parser.add_argument("--endpoint-display-name", default="calls-sales-endpoint")
    args = parser.parse_args()

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.staging_bucket,
    )

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_dir = f"{args.staging_bucket.rstrip('/')}/artifacts/calls-sales/{run_id}"

    training_container_uri = (
        "us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-5:latest"
    )

    job = aiplatform.CustomJob.from_local_script(
        display_name=f"calls-sales-training-{run_id}",
        script_path="src/train_calls_sales.py",
        container_uri=training_container_uri,
        args=["--data-uri", args.data_uri, "--model-dir", model_dir],
        replica_count=1,
        machine_type="n1-standard-4",
    )

    job.run(sync=True)
    print("Training job complete:", job.resource_name)

    model = aiplatform.Model.upload(
        display_name=f"calls-sales-model-{run_id}",
        artifact_uri=model_dir,
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
        ),
        sync=True,
    )

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{args.endpoint_display_name}"'
    )

    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=args.endpoint_display_name,
            sync=True,
        )

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"calls-sales-deployed-{run_id}",
        machine_type="n1-standard-2",
        traffic_percentage=100,
        sync=True,
    )

if __name__ == "__main__":
    main()
