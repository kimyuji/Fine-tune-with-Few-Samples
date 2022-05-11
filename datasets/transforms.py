from torchvision import transforms
import numpy as np

def parse_transform(transform: str, image_size=224, **transform_kwargs):
    if transform == 'RandomColorJitter':
        return transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0)
    elif transform == 'RandomGrayscale':
        return transforms.RandomGrayscale(p=0.1)
    elif transform == 'RandomGaussianBlur':
        return transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5))], p=0.3)
    elif transform == 'RandomCrop':
        return transforms.RandomCrop(image_size)
    elif transform == 'CenterCrop':
        return transforms.CenterCrop(image_size)
    elif transform == 'Resize_up':
        return transforms.Resize(
            [int(image_size * 1.15),
             int(image_size * 1.15)])
    elif transform == 'Normalize':
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif transform == 'Resize':
        return transforms.Resize(
            [int(image_size),
             int(image_size)])
    elif transform == 'RandomRotation':
        return transforms.RandomRotation(degrees=10)

    # check out these two! 
    # elif transform =='RandomHorizontalFlip':
    #     return transforms.RandomHorizontalFlip(p=1.0)
    elif transform == 'RandomResizedCrop':
        return transforms.RandomResizedCrop(image_size)
    
    # New!!
    elif transform == 'RandomAffine':
        return transforms.RandomRotation(degrees=10)


    else:
        method = getattr(transforms, transform)
        return method(**transform_kwargs)


def get_composed_transform(augmentation: str = None, image_size=224) -> transforms.Compose: #return 주석
    if augmentation == 'base':
        transform_list = ['RandomColorJitter', 'RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor',
                          'Normalize']
    elif augmentation == 'strong':
        transform_list = ['RandomResizedCrop', 'RandomColorJitter', 'RandomGrayscale', 'RandomGaussianBlur',
                          'RandomHorizontalFlip', 'ToTensor', 'Normalize']
    elif augmentation is None or augmentation.lower() == 'none':
        transform_list = ['Resize', 'ToTensor', 'Normalize']


    # analyze individually
    elif augmentation == 'randomresizedcrop':
        transform_list = ['RandomResizedCrop', 'ToTensor', 'Normalize']
    elif augmentation == 'randomcoloredjitter':
        transform_list = ['RandomColorJitter', 'ToTensor', 'Normalize', 'Resize']
    elif augmentation == 'randomhorizontalflip':
        transform_list = ['RandomHorizontalFlip', 'Resize', 'ToTensor', 'Normalize']
    elif augmentation == 'randomgrayscale':
        transform_list = ['RandomGrayscale', 'ToTensor', 'Normalize', 'Resize']
    elif augmentation == 'randomgaussianblur':
        transform_list = ['RandomGaussianBlur', 'ToTensor', 'Normalize', 'Resize']
    elif augmentation == 'flipcrop':
        transform_list = ['RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize', 'Resize']

    else:
        raise ValueError('Unsupported augmentation: {}'.format(augmentation))

    transform_funcs = [parse_transform(x, image_size=image_size) for x in transform_list]
    transform = transforms.Compose(transform_funcs)
    return transform





# CutMix
# get 4 corner points of patches 
def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 224
    H = size[3] # 224

    cut_rat = np.sqrt(1. - lam)  # ratio of patch size 
    cut_w = np.int(W * cut_rat)  # patch width
    cut_h = np.int(H * cut_rat)  # patch height
    # square patch

    # 중간 좌표 추출 (uniform sampling)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2