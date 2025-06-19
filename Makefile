ssh:
	@echo "Connecting to TPU VM..."; \
	ZONE=$$(pulumi -s tpu stack output location); \
	gcloud compute tpus tpu-vm ssh --zone $$ZONE graphcast-tpu --project $${GRAPHCAST_PROJECT_ID} -- -L 8081:localhost:8081; \