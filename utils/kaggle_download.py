import os
import zipfile

def download_kaggle_dataset(kaggle_json_path, dataset, output_dir="./dataset"):
    os.makedirs(output_dir, exist_ok=True)
    os.environ["KAGGLE_CONFIG_DIR"] = kaggle_json_path

    print(f"Downloading {dataset}...")
    os.system(f"kaggle datasets download -d {dataset} -p {output_dir}")

    for file in os.listdir(output_dir):
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(output_dir, file), 'r') as z:
                z.extractall(output_dir)
