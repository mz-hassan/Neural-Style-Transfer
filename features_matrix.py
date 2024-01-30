import torch

def get_features(image,model):
  
  layers = {
      '0' : 'conv1_1',
      '5' : 'conv2_1',
      '10' : 'conv3_1',
      '19' : 'conv4_1',
      '21' : 'conv4_2', #content_feature
      '28' : 'conv5_1'
  }

  x = image

  features = {}

  for name, layer in model._modules.items():
    x = layer(x) # gives output from forward pass till layer

    if name in layers: # name eg. (0), (5) ..
      features[layers[name]] = x

  return features

def gram_matrix(tensor):
  b,c,h,w = tensor.size()
  tensor = tensor.view(c, h*w)
  gram = torch.mm(tensor, tensor.t())
  return gram

