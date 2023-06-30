import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_a_train_transform():
    """Get transformer for training data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.HorizontalFlip(p=0.05),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2443, 0.2408, 0.2581)), #normalize
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=0.49, mask_fill_value =None),
        ToTensorV2()])


def get_a_test_transform():
    """Get transformer for test data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.Normalize((0.4942, 0.4851, 0.4504), (0.2439, 0.2402, 0.2582)), #apply normalization
        ToTensorV2()])