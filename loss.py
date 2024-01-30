import torch
from features_matrix import gram_matrix

def content_loss(target_conv4_2, content_conv4_2):
  loss = torch.mean(target_conv4_2 - content_conv4_2)**2

  return loss

def style_loss(style_weights, target_features, style_grams):
  loss = 0

  for layer in style_weights:
    target_f = target_features[layer]
    target_gram = gram_matrix(target_f)
    style_gram = style_grams[layer]
    b,c,h,w = target_f.shape

    layer_loss = style_weights[layer] * torch.mean((target_gram-style_gram)**2)

    loss += layer_loss/(c*h*w)

  return loss

def total_loss(c_loss, s_loss, alpha, beta):
  loss = alpha * c_loss + beta * s_loss
  return loss