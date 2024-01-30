import torch
from torch import optim
from torchvision import models
from loss import total_loss, content_loss, style_loss
from features_matrix import get_features, gram_matrix
from process_image import preprocess, deprocess
import matplotlib.pyplot as plt

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg = models.vgg19(pretrained=True).features
    for parameters in vgg.parameters():
        parameters.requires_grad_(False)
    vgg.to(device)

    content_p = preprocess("images/1690471253952.jpeg").to(device) # content image preprocessed
    style_p = preprocess("images/16e1cea702cf169722364aa427573bb7.jpg").to(device) # style image preprocessed

    content_f = get_features(content_p, vgg) # content features
    style_f = get_features(style_p, vgg) #style features

    style_grams = {layer : gram_matrix(style_f[layer]) for layer in style_f} # style gram matrices for each layer

    target = content_p.clone().requires_grad_(True).to(device)


    alpha = 1 #content reconstruction weight
    beta = 1e5 #style reconstruction weight
    epochs = 300 #3000
    show_every = 50
    style_weights = {
        'conv1_1' : 1.0,
        'conv2_1' : 0.75,
        'conv3_1' : 0.2,
        'conv4_1' : 0.2,
        'conv5_1' : 0.2
    }
    optimizer = optim.Adam([target], lr=0.1) # pixel values of target are updated

    results = []

    for i in range(epochs):
        target_f = get_features(target, vgg)
        c_loss = content_loss(target_f['conv4_2'],content_f['conv4_2'])
        s_loss = style_loss(style_weights,target_f,style_grams)
        t_loss = total_loss(c_loss, s_loss, alpha, beta)

        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()

        if i % show_every == 0:
            print(f"Total loss at epoch {i}: {t_loss}")
            results.append(deprocess(target.detach()))

    plt.figure(figsize = (10,8))
    for i in range(len(results)):
        plt.subplot(int(len(results)/2),2,i+1)
        plt.imshow(results[i])
    plt.show()
    # plt.savefig("updates.jpg")


    target_copy = deprocess(target.detach())
    content_copy = deprocess(content_p)
    style_copy = deprocess(style_p)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (15,5))
    ax1.imshow(content_copy)
    ax1.set_title("Content Image")
    ax2.imshow(style_copy)
    ax2.set_title("Style Image")
    ax3.imshow(target_copy)
    ax3.set_title("Resultant Image")
    fig.savefig("output.jpg")

if __name__ == "__main__":
    train()