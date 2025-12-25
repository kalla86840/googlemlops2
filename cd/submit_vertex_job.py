import argparse
from datetime import datetime, timezone
from google.cloud import aiplatform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-uri", required=True)

    # Use your real staging bucket:
    # gs://inner-domain-397315-vertex-staging-1766657810
    parser.add_argument("--staging-bucket", required=True)

    # Optional naming
    parser.add_argument("--display-name", default="calls-sales-training")
    parser.add_argument("--model-display-name", default="calls-sales-model")
    parser.add_argument("--endpoint-display-name", default="calls-sales-endpoint")

    # Compute choices
    parser.add_argument("--train-machine-type", default="n1-standard-4")
    parser.add_argument("--deploy-machine-type", default="n1-standard-2")

    args = parser.parse_args()

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.staging_bucket,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_dir = f"{args.staging_bucket.rstrip('/')}/model-artifacts/{run_id}"

    # IMPORTANT:
    # Use a Vertex-supported prebuilt training image.
    # You previously hit an "unsupported image" error for some tags.
    # This tag has been working for many people, but if Vertex rejects it,
    # switch to a tag that appears in Vertex docs for your region.
    container_uri = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest"

    print("Using model_dir:", model_dir)
    print("Submitting training job...")

    job = aiplatform.CustomJob.from_local_script(
        display_name=f"{args.display_name}-{run_id}",
        script_path="src/train_calls_sales.py",
        container_uri=container_uri,
        args=[
            "--data-uri", args.data_uri,
            "--model-dir", model_dir,
            "--target-col", "sales",
        ],
        replica_count=1,
        machine_type=args.train_machine_type,
    )

    job.run(sync=True)
    print("✅ Training job complete:", job.resource_name)

    # Upload to Model Registry
    # For sklearn, use the Vertex prebuilt prediction container
    serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"

    print("Uploading model to Vertex Model Registry...")
    model = aiplatform.Model.upload(
        display_name=f"{args.model_display_name}-{run_id}",
        artifact_uri=model_dir,
        serving_container_image_uri=serving_container_image_uri,
        sync=True,
    )
    print("✅ Model uploaded:", model.resource_name)

    # Create or reuse endpoint
    print("Creating (or reusing) endpoint...")
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{args.endpoint_display_name}"'
    )
    if endpoints:
        endpoint = endpoints[0]
        print("Reusing endpoint:", endpoint.resource_name)
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=args.endpoint_display_name,
            sync=True,
        )
        print("✅ Endpoint created:", endpoint.resource_name)

    # Deploy model
    print("Deploying model to endpoint...")
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{args.model_display_name}-deployed-{run_id}",
        machine_type=args.deploy_machine_type,
        traffic_percentage=100,
        sync=True,
    )

    print("✅ Deployment complete!")
    print("Endpoint:", endpoint.resource_name)
    print("Model:", model.resource_name)


if __name__ == "__main__":
    main()
