# pytorch_HDRtransforms
Pytorch transforms based on numpy arrays. I decided to re-write some of the standard pytorch transforms using only numpy to allow for HDR image manipulation.
The file `utils.py` include some methods for loading HDR images in `exr` into numpy arrays and writing numpy arrays into `exr` files

Read `.exr` file:

```python
import utils

# in case of 3-dimensional image
pic3 = utils.read_exr(in_path, ndim=3)

# in case of 1-dimensional image
pic1 = utils.read_exr(in_path, ndim=1)
```

Write `.exr` file:

```python
import utils

# let's assume we have a numpy array called 'pic' with the image stored in the form [HxWxC]
(Rs, Gs, Bs) = [pic[:, :, channel].tostring() for channel in range(pic.shape[-1])]
utils.write_exr(out_path, size=im_size, pixel_values={'R': Rs, 'G': Gs, 'B': Bs})

```

Use transforms in our custom dataset

```python
from torchvision import transforms

import transforms as HDRtransforms

my_transforms = transforms.Compose([
    HDRtransforms.Normalize_01(),
    HDRtransforms.Scale(scale_size),
    HDRtransforms.RandomCrop(crop_size),
    HDRtransforms.RandomHorizontalFlip(),
    HDRtransforms.RandomVerticalFlip(),
    HDRtransforms.ToTensor(),
])
```
