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
    elif transform == 'RandomResizedCrop':
        return transforms.RandomResizedCrop(image_size)
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
    else:
        method = getattr(transforms, transform)
        return method(**transform_kwargs)


def get_composed_transform(augmentation: str = None, image_size=224) -> transforms.Compose:
    if augmentation == 'base':
        transform_list = ['RandomResizedCrop', 'RandomColorJitter', 'RandomHorizontalFlip', 'ToTensor',
                          'Normalize']
    elif augmentation == 'strong':
        transform_list = ['RandomResizedCrop', 'RandomColorJitter', 'RandomGrayscale', 'RandomGaussianBlur',
                          'RandomHorizontalFlip', 'ToTensor', 'Normalize']
    elif augmentation is None or augmentation.lower() == 'none':
        transform_list = ['Resize', 'ToTensor', 'Normalize']
    else:
        raise ValueError('Unsupported augmentation: {}'.format(augmentation))

    transform_funcs = [parse_transform(x, image_size=image_size) for x in transform_list]
    transform = transforms.Compose(transform_funcs)
    return transform

# get 4 corner points of patches
def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    cut_w = np.int(W * cut_rat)  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2