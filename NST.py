import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

model = models.vgg19(pretrained=True).features
total_loss=0
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()

        self.chosen_features=['0','5','10','19','28'] # conv 1-1,conv 2-1,conv 3-1,conv 4-1,conv 5-1
        self.model=models.vgg19(pretrained=True).features[:29]

    def forward(self,x):
        features=[]
        for layer_num , layer in enumerate(self.model):
            x=layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def load_image(image_name):
    image=Image.open(image_name)
    image=loader(image).unsqueeze(0)
    return image.to(device)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size= 224

model=VGG()
loader= transforms.Compose(
    [
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor()
    ]
)


content_img= load_image('Beautiful_roads.jpeg')
original_shape=content_img.shape
resize_back=transforms.Compose(
    [transforms.ToPILImage(),
    transforms.Resize(original_shape[-2],original_shape[-1]),
    transforms.ToTensor()]
)
style_img=load_image('style1.png')

# generated =torch.randn(original_img.shape, device=device,requires_grad=True)
generated=content_img.clone().requires_grad_(True)
# Hyperparameters
total_steps=6001
learning_rate=0.001
alpha=1
beta=0.01
optimizer=optim.Adam([generated],lr=learning_rate)
# For content loss, we chose feature maps from deep layers as it has content information

# For style loss, we choose feature maps from middle layers as initial layers have only  pixel information
# and the feature maps of deeper layers have content info.
# if we choose feature maps from deep layers for style cost function, if there is a cat in style img
# there will be a cat in gen image somewhere, we dont want this to happen.
# so we use feature maps from a layer i.e not too deep nor too shallow

# For style loss: Gram matrix is calculated as follows
# change img shape to --> [num_channels,height*width] and mutiply it with its transpose
for step in range(total_steps):
    gen_features=model(generated)
    content_features=model(content_img)
    style_features=model(style_img)
    content_loss=0
    style_loss=0
    # for i in range(len(gen_features)):
    #     gen=gen_features[i]
    #     content=content_features[i]
    #     style=style_features[i]
    for gen,content,style in zip(
        gen_features,content_features,style_features
    ):
        batch_size, channel, height, width = gen.shape

        a= torch.mean((gen-content)**2)
        content_loss+=a

        # Compute Gram Matrix
        G = gen.view(channel,-1).mm(gen.view(channel,-1).t())
        S = style.view(channel, -1).mm(style.view(channel, -1).t())

        style_loss+=torch.mean((G-S)**2)
    total_loss=alpha*content_loss+beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print('Step {}'.format(step))

    if step%200==0:
        print(total_loss)
    if step%500==0:
        print('Saved image')
        save_image(generated,'generated.png')















