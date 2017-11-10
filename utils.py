import OpenEXR

import Imath
import numpy as np


def read_exr(filepath, ndim=3):
    """
    reads an HDR image stored in a .exr file
    :param filepath:
    :param ndim:
    :return: the HDR image as a numpy array
    """
    # read image and its dataWindow to obtain its size
    pic = OpenEXR.InputFile(filepath)
    dw = pic.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 1:
        channel = pic.channel('R', pt)
        # transform data to numpy
        channel = np.fromstring(channel, dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        return np.array(channel)
    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            channel = pic.channel(c, pt)
            # transform data to numpy
            channel = np.fromstring(channel, dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match numpy style
        return np.array(allchannels).transpose((1, 2, 0))


def write_exr(filepath, size, pixel_values):
    """
    :param filepath:
    :param img_size: tuple with 2 values (height, width)
    :param pixel_values: dict with {"channel name": channel value as string, ...}
    :return:
    """

    # create output file
    out = OpenEXR.OutputFile(filepath, OpenEXR.Header(*size))

    # write pixels
    out.writePixels(pixel_values)
