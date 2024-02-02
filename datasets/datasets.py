from torch.utils.data import Dataset
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from typing import Literal, Tuple, Dict, List
import os


class PixivDataset(Dataset):
    def __init__(self, imagePath: str, imageSize: Literal[256, 512] = 512, transforms: T.Transform | None = None,
                 return_original=True) -> None:
        super().__init__()

        if transforms is not None:
            self.transform = T.Compose(
                [T.ToImage(), T.ToDtype(torch.float32, scale=True),
                 T.Resize([imageSize, imageSize], T.InterpolationMode.BICUBIC, antialias=True), transforms])
        else:
            self.transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                T.Resize([imageSize, imageSize], T.InterpolationMode.BICUBIC, antialias=True),
                transforms
            ])

        self.imagePath = imagePath
        self.imageSize = imageSize
        self.manifest = self.__load_manifest()
        self.return_original = return_original

    def __load_manifest(self) -> List[Dict]:
        import json
        with open(os.path.join(self.imagePath, "manifest.json"), encoding='utf8') as mani_file: manifest = json.load(
            mani_file)
        return manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        data = self.manifest[index]
        image_hr = Image.open(os.path.join(self.imagePath, f'repo-1024/{data["id"]}.jpg'))
        image_lr = self.transform(image_hr)
        if not self.return_original: return image_lr
        image_hr = T.functional.to_tensor(image_hr)
        return image_lr, image_hr
