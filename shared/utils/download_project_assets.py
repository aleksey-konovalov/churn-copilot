import os
import sys
import gdown

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "policies")

# Словарь маршрутизации: куда качать и под каким именем сохранять
ASSETS_MAP = {
    "GOOGLE_DRIVE_MODEL_ID": "data_science/artifacts/best_model.cbm",
    "GOOGLE_DRIVE_TRAIN_ID": "data_science/data/processed/train.csv",
    "GOOGLE_DRIVE_VAL_ID": "data_science/data/processed/val.csv",
    "GOOGLE_DRIVE_TEST_ID": "data_science/data/raw/test.csv",
    "GOOGLE_DRIVE_QDRANT_SNAPSHOT_ID": f"infra/qdrant/snapshots/{QDRANT_COLLECTION}/snapshot.snapshot",
}

def download_assets():
    for env_key, dest_path in ASSETS_MAP.items():
        file_id = os.getenv(env_key)
        if not file_id:
            print(f"[WARNING] {env_key} not set in .env. Skipping {dest_path}.")
            continue

        # Если файл уже скачан или пустой — пропускаем
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"[INFO] Already exists, skipping: {dest_path}")
            continue

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"[INFO] Downloading {env_key} -> {dest_path}")
        
        try:
            # используем quiet=False для прогресс-бара
            gdown.download(id=file_id, output=dest_path, quiet=False, fuzzy=True)
            
            # Проверка на то, что gdown не скачал HTML страницу об ошибке (работает и для csv, и для snapshot)
            if os.path.getsize(dest_path) < 1000 and dest_path.endswith(('.csv', '.snapshot')):
                raise ValueError(f"Downloaded file is too small, possibly an error page from Google Drive.")
                            
        except Exception as e:
            print(f"[ERROR] Failed to download {file_id}: {e}")
            sys.exit(1)

    print("[SUCCESS] All assets downloaded.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    download_assets()
