package main

import (
	tpuv2 "github.com/pulumi/pulumi-google-native/sdk/go/google/tpu/v2"
	"github.com/pulumi/pulumi/sdk/v3/go/pulumi"
	"github.com/pulumi/pulumi/sdk/v3/go/pulumi/config"
)

func main() {
	pulumi.Run(func(ctx *pulumi.Context) error {
		conf := config.New(ctx, "")

		node, err := tpuv2.NewNode(ctx, "examplenodeResourceResourceFromTpuv2", &tpuv2.NodeArgs{
			NodeId:          pulumi.String("graphcast-tpu-esta"),
			Location:        pulumi.String(conf.Require("location")),
			AcceleratorType: pulumi.Sprintf("v5litepod-%d", conf.RequireInt("tpu-chip-core-number")),
			RuntimeVersion:  pulumi.String("v2-tpuv5-litepod"),
			NetworkConfig: tpuv2.NetworkConfigArgs{
				EnableExternalIps: pulumi.Bool(true)},
		})
		if err != nil {
			return err
		}

		ctx.Export("nodeId", node.NodeId)
		ctx.Export("name", node.Name)

		return nil
	})
}

// enable: https://console.cloud.google.com/apis/library/tpu.googleapis.com?project=graphcast-esta&pli=1
// gcloud compute tpus tpu-vm ssh --zone us-central1-a graphcast-tpu-esta --project graphcast-esta -- -L 8081:localhost:8081
