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
    elif transform == 'RandomResizedCrop':
        return transforms.RandomResizedCrop(image_size) # careful,,
    
    # TTA
    elif transform == 'fixed_HFlip':
        return transforms.RandomHorizontalFlip(p=1.0)
    # elif transform == 'fixed_VFlip':
    #     return transforms.RandomVerticalFlip(p=1.0)
    # elif transform == 'fixed_Rotate':
    #     return transforms.RandomRotation(degrees=60)
    elif transform == 'fixed_RCrop':
        return transforms.RandomResizedCrop(image_size, scale = (0.1, 0.9))
    elif transform == 'fixed_CJitter':
        return transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=1.0)
        
    # Aug Intensity
    elif transform == 'RCrop_0.2':
        return transforms.RandomResizedCrop(image_size, scale = (0.2, 0.2))
    elif transform == 'RCrop_0.4':
        return transforms.RandomResizedCrop(image_size, scale = (0.4, 0.4))
    elif transform == 'RCrop_0.6':
        return transforms.RandomResizedCrop(image_size, scale = (0.6, 0.6))
    elif transform == 'RCrop_0.8':
        return transforms.RandomResizedCrop(image_size, scale = (0.8, 0.8))
    
    elif transform == 'Rotate_10':
        return transforms.RandomRotation((10, 10))
    elif transform == 'Rotate_20':
        return transforms.RandomRotation((20, 20))
    elif transform == 'Rotate_30':
        return transforms.RandomRotation((30, 30))
    elif transform == 'Rotate_40':
        return transforms.RandomRotation((40, 40))
    elif transform == 'Rotate_50':
        return transforms.RandomRotation((50, 50))
    elif transform == 'Rotate_60':
        return transforms.RandomRotation((60, 60))

    elif transform == '':
        return
    else:
        method = getattr(transforms, transform)
        return method(**transform_kwargs)

def get_single_transform(augmentation : str):
    transform = [augmentation]
    return transform + ['Resize', 'ToTensor', 'Normalize']

def get_fixed_transform_with_clean(aug_list, image_size=224) -> list: # return list
    transform_list = []
    # first, add original 
    original_transform = ['Resize', 'ToTensor', 'Normalize']
    transform_comp = transforms.Compose([parse_transform(x, image_size=image_size) for x in original_transform])
    transform_list.append(transform_comp)
    for aug in aug_list:
        transform_single = get_single_transform(aug)
        transform_comp = transforms.Compose([parse_transform(x, image_size=image_size) for x in transform_single])
        transform_list.append(transform_comp)
    return transform_list

def get_fixed_transform(aug_list, image_size=224) -> list: # return list
    transform_list = []
    for aug in aug_list:
        transform_single = get_single_transform(aug)
        transform_comp = transforms.Compose([parse_transform(x, image_size=image_size) for x in transform_single])
        transform_list.append(transform_comp)
    return transform_list


def get_composed_transform(augmentation: str = None, image_size=224) -> transforms.Compose: #return 주석
    if augmentation == 'base':
        transform_list = ['RandomColorJitter', 'RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor',
                          'Normalize']
    elif augmentation == 'strong':
        transform_list = ['RandomResizedCrop', 'RandomColorJitter', 'RandomGrayscale', 'RandomGaussianBlur',
                          'RandomHorizontalFlip', 'ToTensor', 'Normalize']

    elif augmentation is None or augmentation.lower() == 'none':
        transform_list = ['Resize', 'ToTensor', 'Normalize'] # Resize필수!TT

    # analyze individually
    elif augmentation == 'randomresizedcrop':
        transform_list = ['RandomResizedCrop', 'ToTensor', 'Normalize']
    elif augmentation == 'randomcolorjitter':
        transform_list = ['RandomColorJitter', 'Resize', 'ToTensor', 'Normalize']
    elif augmentation == 'randomhorizontalflip':
        transform_list = ['RandomHorizontalFlip', 'Resize', 'ToTensor', 'Normalize']
    # elif augmentation == 'randomresizedcrop_fixed':
    #     transform_list = ['RandomResizedCrop_fixed', 'ToTensor', 'Normalize']
    # elif augmentation == 'randomhorizontalflip_fixed':
    #     transform_list = ['RandomHorizontalFlip_fixed', 'Resize', 'ToTensor', 'Normalize']

    # Intensity
    elif augmentation == 'rcrop_0.2':
        transform_list = ['RCrop_0.2', 'ToTensor', 'Normalize']
    elif augmentation == 'rcrop_0.4':
        transform_list = ['RCrop_0.4', 'ToTensor', 'Normalize']
    elif augmentation == 'rcrop_0.6':
        transform_list = ['RCrop_0.6', 'ToTensor', 'Normalize']
    elif augmentation == 'rcrop_0.8':
        transform_list = ['RCrop_0.8', 'ToTensor', 'Normalize']
        
    # Intensity
    elif augmentation == 'rotate_10':
        transform_list = ['Rotate_10', 'Resize', 'ToTensor', 'Normalize']
    elif augmentation == 'rotate_20':
        transform_list = ['Rotate_20', 'Resize', 'ToTensor', 'Normalize']
    elif augmentation == 'rotate_30':
        transform_list = ['Rotate_30', 'Resize', 'ToTensor', 'Normalize']
    elif augmentation == 'rotate_40':
        transform_list = ['Rotate_40', 'Resize', 'ToTensor', 'Normalize']
    elif augmentation == 'rotate_50':
        transform_list = ['Rotate_50', 'Resize', 'ToTensor', 'Normalize']
    elif augmentation == 'rotate_60':
        transform_list = ['Rotate_60', 'Resize', 'ToTensor', 'Normalize']

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

def transforms_ss(input, size=224):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    transform = transforms.Compose([transforms.RandomResizedCrop(size=(size, size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.GaussianBlur(kernel_size=(5,5))])
    return transform(input)
