# pulumi-graphcast

## Log into google CLI

with gsutil

and enable: <https://console.cloud.google.com/apis/library/tpu.googleapis.com?project={GRAPHCAST_PROJECT_ID}&pli=1>

#### upload ssh key

gcloud compute os-login ssh-keys add --key-file=$HOME/.ssh/id_ed25519.pub (adapt path to your ssh key)

get your username by logging into the server via ssh

gcloud compute tpus tpu-vm ssh --zone us-central1-a graphcast-tpu --project {GRAPHCAST_PROJECT_ID} -- -L 8081:localhost:8081

## Set in .env

GRAPHCAST_BUCKET_NAME
GRAPHCAST_PROJECT_ID
