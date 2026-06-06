from ml_phantom.config import DatasetConfig
from ml_phantom.data import load_dataset, save_dataset_npz

config = DatasetConfig(
    base_path="C:/Users/Mahdieh/NK/ML/phantom",
    main_folders=[f"S{i}" for i in range(1, 10)],
)

images, labels, records = load_dataset(config)
print(images.shape)
print(labels.shape)
save_dataset_npz("data/processed/phantom_dataset.npz", images, labels, records)
