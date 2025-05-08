package main

import (
	"crypto/sha1"
	"fmt"
	"log"
	"os"
	"sort"

	"github.com/joho/godotenv"
	"github.com/pulumi/pulumi-command/sdk/go/command/remote"
	iam "github.com/pulumi/pulumi-google-native/sdk/go/google/iam/v1"
	storage "github.com/pulumi/pulumi-google-native/sdk/go/google/storage/v1"
	tpuv2 "github.com/pulumi/pulumi-google-native/sdk/go/google/tpu/v2"
	"github.com/pulumi/pulumi/sdk/v3/go/pulumi"
	"github.com/pulumi/pulumi/sdk/v3/go/pulumi/config"
)

const (
	projectID   = "GRAPHCAST_PROJECT_ID"
	bucketName  = "GRAPHCAST_BUCKET_NAME"
	sshUserName = "SSH_USERNAME"
	sshKeyPath  = "SSH_KEY_PATH"
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
				Metadata: pulumi.StringMap{
					"enable-oslogin": pulumi.String("true"),
				},
			})
			if err != nil {
				return err
			}

			nodeIP := node.NetworkEndpoints.Index(pulumi.Int(0)).AccessConfig().ExternalIp()

			ctx.Export("nodeId", node.NodeId)
			ctx.Export("nodeName", node.Name)
			ctx.Export("nodeIp", nodeIP)

			keyBytes, err := os.ReadFile(os.Getenv(sshKeyPath))
			if err != nil {
				return err
			}

			userName := os.Getenv(sshUserName)

			conn := &remote.ConnectionArgs{
				Host:       nodeIP,
				User:       pulumi.String(userName),
				PrivateKey: pulumi.String(string(keyBytes)),
			}

			// upload all scripts to the server
			scriptsUpload, err := remote.NewCopyToRemote(ctx, "upload-scripts", &remote.CopyToRemoteArgs{
				Connection: conn,
				RemotePath: pulumi.Sprintf("/home/%s/scripts/", userName),
				Source:     pulumi.NewFileArchive("scripts/"),
			}, pulumi.Parent(node))
			if err != nil {
				return err
			}

			// get all file names from the scripts directory
			files, err := os.ReadDir("./scripts")
			if err != nil {
				return err
			}

			// order the files by name
			sort.Slice(files, func(i, j int) bool {
				return files[i].Name() < files[j].Name()
			})

			// execute all files on the server in order
			scriptDependsOn := []pulumi.Resource{scriptsUpload}
			for _, file := range files {
				filePath := fmt.Sprintf("./scripts/%s", file.Name())
				fileContent, err := os.ReadFile(filePath)
				if err != nil {
					return err
				}
				fileHash := fmt.Sprintf("%x", sha1.Sum(fileContent))

				scriptCMD, err := remote.NewCommand(ctx, file.Name(), &remote.CommandArgs{
					Create:     pulumi.String(fmt.Sprintf("chmod +x /home/%s/scripts/%s && /home/%s/scripts/%s", userName, file.Name(), userName, file.Name())),
					Connection: conn,
					Triggers: pulumi.Array{
						pulumi.String(fileHash),
					},
				},
					pulumi.Parent(scriptsUpload),
					pulumi.DependsOn(scriptDependsOn),
				)
				if err != nil {
					return err
				}
				scriptDependsOn = append(scriptDependsOn, scriptCMD)
			}

			// upload requirements.txt to the server
			requirementsUpload, err := remote.NewCopyToRemote(ctx, "upload-requirements", &remote.CopyToRemoteArgs{
				Connection: conn,
				RemotePath: pulumi.Sprintf("/home/%s/requirements/", userName),
				Source:     pulumi.NewFileArchive("requirements/"),
			}, pulumi.Parent(node))
			if err != nil {
				return err
			}

			_, err = remote.NewCommand(ctx, "pip install", &remote.CommandArgs{
				Create: pulumi.String(`
source .venv/bin/activate
pip install -r requirements/server.txt`),
				Connection: conn,
				Triggers: pulumi.Array{
					requirementsUpload,
				},
			},
				pulumi.Parent(requirementsUpload),
				pulumi.DependsOn(scriptDependsOn),
				pulumi.DependsOn([]pulumi.Resource{requirementsUpload}),
			)
			if err != nil {
				return err
			}

			_, err = remote.NewCopyToRemote(ctx, "upload-dot-env", &remote.CopyToRemoteArgs{
				Connection: conn,
				RemotePath: pulumi.Sprintf("/home/%s/.env", userName),
				Source:     pulumi.NewFileAsset(".env"),
			}, pulumi.Parent(node))
			if err != nil {
				return err
			}
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
		}

		return nil
	})
}

// gcloud compute tpus tpu-vm ssh --zone us-central1-a graphcast-tpu --project graphcast-esta -- -L 8081:localhost:8081
