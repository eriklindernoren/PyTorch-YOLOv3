import imgaug.augmenters as iaa
from torchvision import transforms
from pytorchyolo.utils.transforms import ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug


class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            # rotate by -45 to 45 degrees (affects segmaps)
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1)),
            iaa.AddToBrightness((-10, 10)),
            iaa.AddToHue((-5, 5)),
            iaa.Fliplr(0.5),
        ])


AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])
