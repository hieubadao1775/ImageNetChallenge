from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import Compose


class Animal(Dataset):
    def __init__(self, root, transform=Compose([])):
        super().__init__()
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.categories = sorted(os.listdir(root))

        category_paths = [os.path.join(root, category) for category in self.categories]
        for i, category_path in enumerate(category_paths):
            image_files = os.listdir(category_path)
            self.image_paths.extend([os.path.join(category_path, image_file) for image_file in image_files])
            self.labels.extend([i] * len(image_files))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        label = self.labels[item]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label



if __name__ == "__main__":
    data = Animal(root="../data/animals")
    index = 12355
    image, label = data.__getitem__(index)

    image.show()
    print(label)