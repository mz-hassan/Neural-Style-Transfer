from PIL import Image
import numpy as np
from torchvision import transforms as T

def preprocess(img_path, max_size = 500):
  image = Image.open(img_path).convert('RGB')

  if max(image.size) > max_size: # if size is greater than max_size, resize to maxsize
    size = max_size
  else:
    size = max(image.size)

  img_transforms = T.Compose([
      T.Resize(size),
      T.ToTensor(), #(224, 224, 3) -> (3, 224, 224)
      T.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
  ])

  image = img_transforms(image)

  image = image.unsqueeze(0) # first dimension should be batch size
  # (3, 224, 224) -> (1, 3, 224, 224)

  return image

    
def deprocess(tensor):
    image = tensor.to('cpu').clone()

    image = image.numpy()
    image = image.squeeze(0) #(1, 3, 224, 224) -> (3, 224, 224)
    image = image.transpose(1,2,0) #(3, 224, 224) -> (224, 224, 3)

    image = image*np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0,1)

    return image