from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms.v2 as T, torchvision.transforms.v2.functional as TF
from typing import Literal, Tuple, Dict, List
import os
from PIL import Image


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
        

        return TF.pad(TF.to_dtype(TF.to_image(new_im), dtype=torch.float32, scale=True), padding=[
            (SIZE - new_size[0]) >> 1,  #left
            (SIZE - new_size[1]) >> 1,  #top
            (SIZE - new_size[0]) - ((SIZE - new_size[0]) >> 1),  #right
            (SIZE - new_size[1]) - ((SIZE - new_size[1]) >> 1),  #bottom            
        ], padding_mode='edge')



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
