package main

import (
	"log"
	"os"

	"github.com/joho/godotenv"
	iam "github.com/pulumi/pulumi-google-native/sdk/go/google/iam/v1"
	storage "github.com/pulumi/pulumi-google-native/sdk/go/google/storage/v1"
	tpuv2 "github.com/pulumi/pulumi-google-native/sdk/go/google/tpu/v2"
	"github.com/pulumi/pulumi/sdk/v3/go/pulumi"
	"github.com/pulumi/pulumi/sdk/v3/go/pulumi/config"
)

const (
	projectID  = "GRAPHCAST_PROJECT_ID"
	bucketName = "GRAPHCAST_BUCKET_NAME"
)

func main() {
	pulumi.Run(func(ctx *pulumi.Context) error {
		conf := config.New(ctx, "")

		// Load .env file
		err := godotenv.Load()
		if err != nil {
			log.Fatal("Error loading .env file")
		}

		if conf.GetBool("enable_tpu") {

			// Create a TPU node
			node, err := tpuv2.NewNode(ctx, "tpu", &tpuv2.NodeArgs{
				NodeId:          pulumi.String("graphcast-tpu"),
				Location:        pulumi.String(conf.Require("location")),
				AcceleratorType: pulumi.Sprintf("v5litepod-%d", conf.RequireInt("tpu-chip-core-number")),
				RuntimeVersion:  pulumi.String("v2-tpuv5-litepod"),
				NetworkConfig: tpuv2.NetworkConfigArgs{
					EnableExternalIps: pulumi.Bool(true),
				},
				Project: pulumi.String(os.Getenv(projectID)),
			})
			if err != nil {
				return err
			}

			ctx.Export("nodeId", node.NodeId)
			ctx.Export("nodeName", node.Name)
		}

		if conf.GetBool("enable_bucket") {

			// Create a GCS bucket
			bucket, err := storage.NewBucket(ctx, "bucket", &storage.BucketArgs{
				Name:     pulumi.String(os.Getenv(bucketName)),
				Location: pulumi.String(conf.Require("location")),
				Project:  pulumi.String(os.Getenv(projectID)),
			})
			if err != nil {
				return err
			}

			// Create a Cloud TPU service account
			serviceAccount, err := iam.NewServiceAccount(ctx, "tpuServiceAccount", &iam.ServiceAccountArgs{
				AccountId:   pulumi.String("tpu-service-account"),
				DisplayName: pulumi.String("TPU Service Account"),
			})
			if err != nil {
				return err
			}

			// Grant the service account write permissions on the bucket
			_, err = storage.NewBucketIamMember(ctx, "BucketIamBinding", &storage.BucketIamMemberArgs{
				Name:   bucket.Name,
				Role:   pulumi.String("roles/storage.objectCreator"),
				Member: pulumi.Sprintf("serviceAccount:%s", serviceAccount.Email),
			})
			if err != nil {
				return err
			}

			ctx.Export("bucketName", bucket.Name)
			ctx.Export("serviceAccountEmail", serviceAccount.Email)

			// mount the bucket
		}

		return nil
	})
}

// gcloud compute tpus tpu-vm ssh --zone us-central1-a graphcast-tpu-esta --project graphcast-esta -- -L 8081:localhost:8081
