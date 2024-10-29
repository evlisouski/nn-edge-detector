from torch.utils.data import Dataset
from PIL import Image


class BDSD500Dataloader(Dataset):
    def __init__(self, file_path, transform=None):
        self.paths = list(self._load_paths_from_file(file_path))
        self.transform = transform

    def _load_paths_from_file(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                path_tmp = line.strip().split()
                if len(path_tmp) >= 2:  # Убедимся, что есть хотя бы два пути
                    yield "datasets/BSDS500/" + path_tmp[0], "datasets/BSDS500/" + path_tmp[1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):        
        source_image_path, target_image_path = self.paths[idx]
        # print(source_image_path)

        source_image = Image.open(source_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("L")  # Границы в черно-белом формате

        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return source_image, target_image