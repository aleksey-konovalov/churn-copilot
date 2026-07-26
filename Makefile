# Автоматически вычисляем UID/GID хост-пользователя
UID := $(shell id -u)
GID := $(shell id -g)
export UID GID

.PHONY: download-assets
download-assets:
	@echo "Downloading model, datasets, and Qdrant artifacts..."
	@docker compose run --rm dev-tools \
		bash -c "pip install --target=/tmp/packages 'gdown>=5.1.0,<6' python-dotenv && \
				 python shared/utils/download_project_assets.py || \
				 (echo 'Не получилось скачать файлы.' && exit 1)"
	@echo "Файлы успешно скачаны."


# Цель для восстановления Qdrant из снапшота:
.PHONY: restore-qdrant
restore-qdrant: download-assets
	@echo "Restoring Qdrant from snapshot..."
	@COLLECTION_NAME=$$(grep '^QDRANT_COLLECTION=' shared/.env | cut -d '=' -f2); \
	SNAPSHOT_DIR=infra/qdrant/snapshots/$$COLLECTION_NAME; \
	echo "Using collection: $$COLLECTION_NAME"; \
	CONTAINER_ID=$$(docker compose ps -q qdrant); \
	WAS_RUNNING=true; \
	if [ -z "$$CONTAINER_ID" ] || [ -z "$$(docker inspect -f '{{.State.Running}}' $$CONTAINER_ID 2>/dev/null)" ]; then \
		echo "Qdrant container is not running. Starting it..."; \
		docker compose up -d qdrant; \
		CONTAINER_ID=$$(docker compose ps -q qdrant); \
		WAS_RUNNING=false; \
	fi; \
	echo "Waiting for Qdrant to become ready..."; \
	RETRIES=30; \
	until curl -sS http://localhost:3030/healthz > /dev/null 2>&1; do \
		RETRIES=$$((RETRIES - 1)); \
		if [ $$RETRIES -le 0 ]; then \
			echo "Qdrant failed to start within the timeout." && exit 1; \
		fi; \
		sleep 1; \
	done; \
	echo "Qdrant is ready."; \
	if [ ! -f "$$SNAPSHOT_DIR/snapshot.snapshot" ]; then \
		echo "Snapshot file not found at $$SNAPSHOT_DIR/snapshot.snapshot" && exit 1; \
	fi; \
	CONTAINER_NAME=$$(docker inspect --format '{{.Name}}' $$CONTAINER_ID | sed 's/^\///'); \
	echo "Container name: $$CONTAINER_NAME"; \
	curl -sS -X DELETE "http://localhost:3030/collections/$$COLLECTION_NAME" > /dev/null 2>&1; \
	echo "Copying snapshot directly into container via docker cp..."; \
	docker exec "$$CONTAINER_NAME" mkdir -p /qdrant/snapshots/$$COLLECTION_NAME; \
	docker cp "$$SNAPSHOT_DIR/snapshot.snapshot" "$$CONTAINER_NAME:/qdrant/snapshots/$$COLLECTION_NAME/snapshot.snapshot"; \
	docker exec "$$CONTAINER_NAME" chmod 644 /qdrant/snapshots/$$COLLECTION_NAME/snapshot.snapshot; \
	echo "Recreating collection directly from snapshot file..."; \
	RESTORE_RESPONSE=$$(curl -sS -X PUT "http://localhost:3030/collections/$$COLLECTION_NAME?snapshot=/qdrant/snapshots/$$COLLECTION_NAME/snapshot.snapshot" \
		-H "Content-Type: application/json" \
		-d '{"vectors": {"size": 1536, "distance": "Cosine"}}'); \
	echo "Restore initiated response: $$RESTORE_RESPONSE"; \
	if ! echo "$$RESTORE_RESPONSE" | grep -q '"status":"ok"'; then \
		echo "Restore might have failed. Please check Qdrant container logs." && exit 1; \
	fi; \
	if [ "$$WAS_RUNNING" = "false" ]; then \
		echo "Stopping Qdrant container (was started by this script)..."; \
		docker compose stop qdrant; \
	fi; \
	echo "Qdrant restored successfully from snapshot."					

# Цель для создания снапшота текущей коллекции Qdrant:
.PHONY: save-snapshot
save-snapshot:
	@echo "Creating Qdrant snapshot..."
	@COLLECTION_NAME=$$(grep '^QDRANT_COLLECTION=' shared/.env | cut -d '=' -f2); \
	SNAPSHOT_DIR=infra/qdrant/snapshots/$$COLLECTION_NAME; \
	echo "Using collection: $$COLLECTION_NAME"; \
	CONTAINER_NAME=$$(docker compose ps -q qdrant); \
	if [ -z "$$CONTAINER_NAME" ]; then \
		echo "Qdrant container is not running. Please start it with 'docker compose up qdrant -d'." && exit 1; \
	fi; \
	CREATE_RESPONSE=$$(curl -sS -X POST "http://localhost:3030/collections/$$COLLECTION_NAME/snapshots"); \
	SNAPSHOT_NAME=$$(echo "$$CREATE_RESPONSE" | grep -o '"name":"[^"]*"' | grep -o '[^"]*\.snapshot' | head -n 1); \
	if [ -z "$$SNAPSHOT_NAME" ]; then \
		echo "Failed to create snapshot. Response: $$CREATE_RESPONSE" && exit 1; \
	fi; \
	echo "Snapshot created: $$SNAPSHOT_NAME"; \
	mkdir -p $$SNAPSHOT_DIR; \
	docker cp $$CONTAINER_NAME:/qdrant/snapshots/$$COLLECTION_NAME/$$SNAPSHOT_NAME $$SNAPSHOT_DIR/snapshot.snapshot; \
	if [ $$? -ne 0 ]; then \
		echo "Failed to copy snapshot from container. Please check Qdrant container name and paths." && exit 1; \
	fi; \
	echo "Snapshot saved successfully to $$SNAPSHOT_DIR/snapshot.snapshot"