import argparse
from google.cloud import aiplatform

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--data-uri", required=True)
    p.add_argument("--staging-bucket", required=True)  # e.g. gs://... (NO trailing slash needed)
    args = p.parse_args()

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.staging_bucket,
    )

    # Vertex prebuilt training container (must be Vertex-offered for python package training)
    container_uri = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest"

    # This is the critical part:
    # base_output_dir causes Vertex to pass --model-dir to the runner.
    base_output_dir = f"{args.staging_bucket}/model-artifacts"

    job = aiplatform.CustomJob.from_local_script(
        display_name="calls-sales-training",
        script_path="src/train_calls_sales.py",
        container_uri=container_uri,
        args=[
            "--data-uri", args.data_uri,
            # Do NOT add --model-dir here; Vertex will inject it when base_output_dir is set.
        ],
        replica_count=1,
        machine_type="n1-standard-4",
        base_output_dir=base_output_dir,
    )

    job.run(sync=True)
    print("Vertex job complete:", job.resource_name)

if __name__ == "__main__":
    main()
