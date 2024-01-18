import skimage
import typing
import numpy as np


class CoOccurenceMatrix:
    def __init__(
            self,
            distances: [int],
            angles: [float],
            levels: int,
            channel: typing.Literal[
                "grey",
                "red",
                "green",
                "blue",
                "every"
            ],
    ) -> None:
        """
        :param distances: Distance between pair pixels
        :param angles: Angle between pair pixels
        :param levels: Number of levels for each pixel given by 2**levels
        :param channel: Which channel to use for the co-occurence matrix must be one of ["grey", "red", "green", "blue", "every"]
        """
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.channel = channel

    def __call__(self, image: np.ndarray):

        if "every" in self.channel:
            assert len(image.shape) >= 3
            assert self.levels % 3 == 0

            image = image // 2 ** (self.levels // 3)

            one_channel_image = 2 ** (2 * self.levels // 3) * image[:, :, 0]  # R
            one_channel_image += 2 ** (self.levels // 3) * image[:, :, 1]  # G
            one_channel_image += image[:, :, 2]  # B

            return skimage.feature.graycomatrix(
                one_channel_image,
                self.distances,
                self.angles,
                self.levels,
                normed=True,
            )

        elif "grey" in self.channel:

            # If the image is not grey
            if len(image.shape) >= 3:
                image = skimage.color.rgb2gray(image)
                image = skimage.img_as_ubyte(image)

            image = image // 2 ** (8 - self.levels)
            return skimage.feature.graycomatrix(
                image,
                self.distances,
                self.angles,
                self.levels,
                normed=True,
            )
        else:
            channels = {
                "red": 0,
                "green": 1,
                "blue": 2,
            }

            # Select the correct channel
            image = image[:, :, channels[self.channel]]
            image = image // 2 ** (8 - self.levels)

            return skimage.feature.graycomatrix(
                image,
                self.distances,
                self.angles,
                self.levels,
                normed=True,
            )
