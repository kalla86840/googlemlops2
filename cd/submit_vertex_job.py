import argparse
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

    container_uri = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest"

    job = aiplatform.CustomJob.from_local_script(
        display_name="calls-sales-training",
        script_path="src/train_calls_sales.py",
        container_uri=container_uri,
        args=[
            "--data-uri", args.data_uri
        ],
        replica_count=1,
        machine_type="n1-standard-4",

        # 🔴 THIS LINE IS REQUIRED
        base_output_dir=f"{args.staging_bucket}/model-artifacts",
    )

    job.run(sync=True)
    print("Vertex job complete:", job.resource_name)

if __name__ == "__main__":
    main()
