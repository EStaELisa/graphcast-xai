pulumi up

# log into the server, adapt location if neccessary (or make ssh)
gcloud compute tpus tpu-vm ssh --zone us-central1-a graphcast-tpu --project graphcast-esta -- -L 8081:localhost:8081
# gcloud compute tpus tpu-vm ssh --zone us-south1-a graphcast-tpu --project graphcast-esta -- -L 8081:localhost:8081

# start jupityer lab kernel without encrypted token
.venv/bin/python3 -m notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8081 --NotebookApp.port_retries=0 --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True --no-browser
# Notebook URL:  http://127.0.0.1:8081/tree