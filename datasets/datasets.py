from torch.utils.data import Dataset, IterableDataset
import torch
import torchvision
from glob import glob
import pandas as pd
import torchvision.transforms.v2 as T, torchvision.transforms.v2.functional as TF
from typing import Literal, Tuple, Dict, List
import os
from PIL import Image
from io import BytesIO


class ResizeAndPad(T.Transform):
    def __init__(self, size:int|Tuple[int, int]):
        """
        Custom transform which performs a resize, and then pads either both left and right, or both top and bottom of
        the image to match the given size, gives `torch.Tensor`, scaled down to range (0, 1) as output
        :param size: The size desired for the image to be resized and padded
        """
        super().__init__()
        self.size = size

    def forward(self, im:Image.Image) -> torch.Tensor:
        """
        Resizes and pads the image.

        :param im: A PIL.Image, the image of which resizing and padding is desired
        :return: The tensor representing the image, resized and padded to specified shape with values scaled down to
            (0, 1).
        """
        SIZE = self.size
        ratio = SIZE / max(im.size)

        new_size = tuple([int(x*ratio) for x in im.size])

        new_im = im.resize(new_size, Image.LANCZOS)

        padding = [(SIZE - new_size[0]) // 2,
                    (SIZE - new_size[1]) // 2,
                    (SIZE - new_size[0]) - (SIZE - new_size[0]) // 2,
                    (SIZE - new_size[1]) - (SIZE - new_size[1]) // 2,
                ]
        
        scaled = TF.to_dtype(TF.to_image(new_im), dtype=torch.float32, scale=True)
        for i in range(4):
            while padding[i] >= new_size[i%2]:
                pad = [0, 0, 0, 0]
                pad[i] = new_size[i%2] - 1
                scaled = TF.pad(scaled, padding=pad, padding_mode='reflect')
                padding[i] -= new_size[i%2] - 1


        return TF.pad(scaled,
                      padding=padding, padding_mode='reflect')



class PixivDataset(Dataset):
    def __init__(self, imagePath: str, imageSize: Literal[256, 512] = 256, transforms: T.Transform | None = None,
                 return_original=True, resize_rate:float=.7) -> None:
        super().__init__()
        assert resize_rate > 0 and resize_rate < 1
        if transforms is not None:
            self.transform = T.Compose(
                [T.RandomChoice([ResizeAndPad(imageSize), 
                                 T.Compose([T.RandomCrop(imageSize, pad_if_needed=True, padding_mode='edge'), T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                                 T.Compose([T.Resize(imageSize), T.CenterCrop(imageSize), T.ToImage(), T.ToDtype(torch.float32, scale=True)])], p=[resize_rate, (1-resize_rate)/2, (1-resize_rate)/2]), transforms])
        else:
            self.transform = T.RandomChoice([ResizeAndPad(imageSize), 
                                 T.Compose([T.RandomCrop(imageSize, pad_if_needed=True, padding_mode='edge'), T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                                 T.Compose([T.Resize(imageSize), T.CenterCrop(imageSize), T.ToImage(), T.ToDtype(torch.float32, scale=True)])],
                                 p=[resize_rate, (1-resize_rate)/2, (1-resize_rate)/2])

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
        image_hr = Image.open(os.path.join(self.imagePath, f'repo/{data["id"]}.webp'))
        image_lr = self.transform(image_hr)
        if not self.return_original: return image_lr
        return image_lr, image_hr



# This also comes with captions
class DBParquetDataset(IterableDataset):
    def __init__(self, dataset_path:str, split:Literal[0,1,2,3]=0,
                 imageSize: Literal[256, 512]|None = 256, transforms: T.Transform | None = None,
                 resize_rate:float=.7) -> None:
        """
        `dataset_path`:
        - path to the directory containing the parquet files.
        - will read all parquet files inside the path, so be as precise as possible.

        `imageSize`:
        - either 256 or 512 for 256x256 or 512x512 images
        - None for full size

        `split`:
        - 0: train split 256x256
        - 1: valid split 256x256
        - 2: train split full
        - 3: valid split full

        `transforms`:
        - applies after all default transforms (resize/pad or center crop or random crop -> to tensor -> transforms)
        - can be set (and default) to None for no following transforms

        `resize_rate`:
        - only applicable if image size is specified (256 or 512)
        - the probability of the image being resized and padded to specified size rather than center/random cropping.
        - a float from 0 to 1 (exclusive)
        """
        super().__init__()
        assert resize_rate > 0 and resize_rate < 1
        if imageSize is not None:
            if transforms is not None:
                self.transform = T.Compose(
                    [T.RandomChoice([ResizeAndPad(imageSize), 
                                    T.Compose([T.RandomCrop(imageSize, pad_if_needed=True, padding_mode='edge'), T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                                    T.Compose([T.Resize(imageSize), T.CenterCrop(imageSize), T.ToImage(), T.ToDtype(torch.float32, scale=True)])], p=[resize_rate, (1-resize_rate)/2, (1-resize_rate)/2]), transforms])
            else:
                self.transform = T.RandomChoice([ResizeAndPad(imageSize), 
                                    T.Compose([T.RandomCrop(imageSize, pad_if_needed=True, padding_mode='edge'), T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                                    T.Compose([T.Resize(imageSize), T.CenterCrop(imageSize), T.ToImage(), T.ToDtype(torch.float32, scale=True)])],
                                    p=[resize_rate, (1-resize_rate)/2, (1-resize_rate)/2])
        else:
            self.transform = T.Compose([
                T.ToImage(), T.ToDtype(torch.float32, scale=True),
                transforms or T.Identity()
            ])
        self.imagePath = dataset_path
        self.split = split
        self.imageSize = imageSize
        self.manifest = self.__get_files()

    def __get_files(self) -> List[str]:
        all = sorted(glob(f"{self.imagePath}/**/*.parquet", recursive=True), key=lambda x: x.split('/')[-1])
        if self.split == 0:
            return all[:-104]
        if self.split == 1:
            return all[-104:-100]
        if self.split == 2:
            return all[-100:-2]
        return all[-2:]

    def parse_parquet(self, file_path):
        """
        Generator that yields rows from a Parquet file.
        
        Args:
            file_path (str): Path to the Parquet file.
        """
        df = pd.read_parquet(file_path)
        for _, row in df.iterrows():
            iofile = BytesIO(row['image']['bytes'])
            image = Image.open(iofile)
            cap = row['tags']
            yield (self.transform(image), cap)

    def get_stream(self, file_paths=None):
        """
        Generator that iterates through all files and yields rows.
        """
        if file_paths is None:
            file_paths = self.manifest
        for file_path in file_paths:
            yield from self.parse_parquet(file_path)

    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.get_stream()
        total_workers = worker_info.num_workers
        worker_id = worker_info.id
        # Distribute file paths among workers
        files_per_worker = len(self.manifest) // total_workers
        extra = len(self.manifest) % total_workers
        start = worker_id * files_per_worker + min(worker_id, extra)
        end = start + files_per_worker + (1 if worker_id < extra else 0)
        return self.get_stream(self.manifest[start:end])

