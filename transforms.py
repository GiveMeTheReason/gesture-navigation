import torchvision.transforms as transforms

target_size = (72, 96)

nearest = transforms.InterpolationMode.NEAREST

class Normalize():
    def __call__(self, tensor):
        mask = tensor > 0
        tensor_min = tensor[mask].min()
        tensor_max = tensor.max()
        tensor[mask] = 1 - (tensor[mask] - tensor_min) / (tensor_max - tensor_min)
        return tensor


class Train_Transforms():
    def __init__(self):
        self.transforms = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Resize(target_size, interpolation=nearest),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=60, translate=(0.3, 0.3), scale=(0.7, 1.3), interpolation=nearest),
                transforms.RandomPerspective(distortion_scale=0.4, p=0.5, interpolation=nearest),
                Normalize()
            ])
    
    def __call__(self, tensor):
        return self.transforms(tensor)


class Test_Transforms():
    def __init__(self):
        self.transforms = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Resize(target_size, interpolation=nearest),
                Normalize()
            ])
    
    def __call__(self, tensor):
        return self.transforms(tensor)
