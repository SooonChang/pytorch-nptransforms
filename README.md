# Pytorch np_transforms
Pytorch transforms based on numpy arrays. I decided to re-write some of the standard pytorch transforms using only numpy operations that allow for High Dynamic Range image manipulation.
The file `exr_data.py` include some methods for loading HDR images in `exr` format into numpy arrays and writing numpy arrays into `exr` files.

A list of implemented transforms based on numpy arrays are:
 - Bilateral Filter
 - Median Filter
 - Image Rotation (either randomly sample an angle between two bounds or with a fixed angle)
 - Random Horizontal Flip
 - Random Vertical Flip
 - Random Crop 
 - Center Crop
 - Five Crops (4 courners + center)
 - Normalize 0-1 (Normalize the image between 0-1)
 - Random Erasing (Random Erasing Data Augmentation by Zhong et al.)
 - To Tensor
 - rgb2xyz (Change in the color space)
 - xyz2rgb (The opposite change in color space)
 - Lambda (Based on torchvision.transforms.Lambda)
 - Compose (Based on torchvision.transforms.Compose)
 - Normalize (Based on torchvision.transforms.Normalize)

## Dependencies

- `numpy`
- `torch` http://pytorch.org/
- `torchvision` http://pytorch.org/
- `scipy` 
- (Only if you want to load/ save HDR images) `OpenEXR` http://www.excamera.com/sphinx/articles-openexr.html



## Usage examples

Create a dataset that loads hdr images in `.exr` format:

```python
import exr_data

trf = np_transforms.Compose([
    np_transforms.Scale(size=(256, 256)),
    np_transforms.RandomCrop(size=(224, 224)),
    np_transforms.RandomVerticalFlip(prob=0.5),
    np_transforms.RandomHorizontalFlip(prob=0.5),
    np_transforms.RotateImage(angles=(-15, 15)),
    np_transforms.ToTensor(),
])

data_train = exr_data.exrData(root=os.path.join(ROOT_DIR, 'train'),
                                  loader=exr_data.exr_loader,
                                  transform=trf)

                                  
```

Write a numpy array into a `.exr` file:

```python
import exr_data

# let's assume we have a numpy array called 'pic' with the image stored in the form [HxWxC]

(Rs, Gs, Bs) = [pic[:, :, channel].tostring() for channel in range(pic.shape[-1])]
exr_data.exr_writer(out_path, size=im_size, pixel_values={'R': Rs, 'G': Gs, 'B': Bs})

```
