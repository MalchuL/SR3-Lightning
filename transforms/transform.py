import torchvision.transforms as transforms
from PIL import Image
import albumentations as A

# just modify the width and height to be multiple of 4
from albumentations.augmentations.functional import center_crop
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import Lambda
import cv2

def __adjust(img):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __scale_width(img, target_width):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


class StagedTransform(transforms.Compose):


    def get_alb_transform(self, alb_transforms):
        composed = A.Compose(alb_transforms, p=1)
        alb_transform = [Lambda(lambda x: composed(image=x)['image'])]

        return transforms.Compose(alb_transform)


    def __init__(self, pre_transform, strong_transform=[], post_transform=[]):
        self.pre_transform = self.get_alb_transform(pre_transform)
        self.strong_transform = self.get_alb_transform(strong_transform)
        self.post_transform = self.get_alb_transform(post_transform)

        super().__init__([self.get_alb_transform(pre_transform + strong_transform + post_transform)])





def get_infer_transform(max_size=384, must_divied=None):
    transform_list = []
    pre_process = [
        A.SmallestMaxSize(max_size, always_apply=True, interpolation=cv2.INTER_CUBIC),
        #A.CenterCrop(max_size, max_size, always_apply=True)
        ]

    if must_divied:
        class DividedResize(A.ImageOnlyTransform):
            def __init__(self, divided):
                super().__init__(always_apply=True)
                self.divided = divided

            def apply(self, img, **params):
                h, w, _ = img.shape
                h = h // must_divied * must_divied
                w = w // must_divied * must_divied

                img = center_crop(img, h, w)
                return img


        pre_process.append(DividedResize(must_divied))
    post_process = [A.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5)),
                    ToTensorV2()]

    composed = pre_process + post_process

    composed = A.Compose(composed, p=1)

    transform_list += [Lambda(lambda x: composed(image=x)['image']),
                   ]
    return transforms.Compose(transform_list)


def get_cartoon_transform(opt, isTrain):
    very_rare_prob = 0.05
    rare_prob = 0.1
    medium_prob = 0.2
    normal_prob = 0.3
    often_prob = 0.6
    compression_prob = 0.35


    if isTrain:
        pre_process = [
            A.ShiftScaleRotate(shift_limit=0.001, rotate_limit=20, scale_limit=0.3, interpolation=cv2.INTER_CUBIC,
                               p=normal_prob),
            A.SmallestMaxSize(opt.loadSize, always_apply=True, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(opt.fineSize, opt.fineSize, always_apply=True)]
    else:
        pre_process = [
            A.SmallestMaxSize(opt.loadSize, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(opt.fineSize, opt.fineSize, always_apply=True)]


    strong = []

    post_process = [A.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5)),
                    ToTensorV2()]

    if isTrain:
        composed = pre_process + strong + post_process
    else:
        composed = pre_process + post_process
        strong = []

    return StagedTransform(pre_process, strong, post_process)

def get_transform(opt, isTrain, use_blur=False):
    very_rare_prob = 0.05
    rare_prob = 0.1
    medium_prob = 0.2
    normal_prob = 0.3
    often_prob = 0.6
    compression_prob = 0.35

    transform_list = []
    if isTrain:
        pre_process = [
            A.ShiftScaleRotate(shift_limit=0.001, rotate_limit=20, scale_limit=0.3, interpolation=cv2.INTER_CUBIC,
                               p=normal_prob),
            A.SmallestMaxSize(opt.loadSize, always_apply=True, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(opt.fineSize, opt.fineSize, always_apply=True)]
    else:
        pre_process = [
            A.SmallestMaxSize(opt.loadSize, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(opt.fineSize, opt.fineSize, always_apply=True)]

    if use_blur:
        pre_process += [A.GaussianBlur(blur_limit=5, always_apply=True)]


    strong = [


        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=normal_prob),
            A.MotionBlur(p=rare_prob),
            A.Downscale(scale_min=0.6, scale_max=0.8, interpolation=cv2.INTER_CUBIC, p=rare_prob),
        ], p=normal_prob),

        A.OneOf([
            A.ToGray(p=often_prob),
            A.ToSepia(p=very_rare_prob)
        ], p=very_rare_prob),

        A.OneOf([
            A.ImageCompression(quality_lower=39, quality_upper=60, p=compression_prob),

            A.MultiplicativeNoise(multiplier=[0.92, 1.08], elementwise=True, per_channel=True, p=compression_prob),
            A.ISONoise(p=compression_prob)
        ], p=compression_prob),
        A.OneOf([
            A.CLAHE(p=normal_prob),
            A.Equalize(by_channels=False, p=normal_prob),
            A.RGBShift(p=normal_prob),
            A.HueSaturationValue(p=normal_prob),
            A.RandomBrightnessContrast(p=normal_prob),
            # A.RandomShadow(p=very_rare_prob, num_shadows_lower=1, num_shadows_upper=1,
            #               shadow_dimension=5, shadow_roi=(0, 0, 1, 0.5)),
            A.RandomGamma(p=normal_prob),
        ]),


    ]

    post_process = [A.Normalize((0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)),
        ToTensorV2()]


    if isTrain:
        composed = pre_process + strong + post_process
    else:
        composed = pre_process + post_process
        strong = []


    return StagedTransform(pre_process, strong, post_process)