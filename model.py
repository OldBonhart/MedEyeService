import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import skimage
from skimage.transform import resize
from skimage import img_as_ubyte

import numpy as np

from PIL import Image
import matplotlib as mpl
from matplotlib import cm

## Uploading model
resnet18 = torchvision.models.resnet18(pretrained=False)
resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 5)
f = open('aptos_resnet18.pt', 'rb')
resnet18.load_state_dict(torch.load(f, map_location='cpu'))
resnet18 = resnet18.cpu()

# Save features from model
class SaveFeatures():
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()


# Image preprocessing and make prediction
class MakePredict():

    def PIL2array(self, img):
        print('------------------------', img)
        return np.array(img.getdata(),
                        np.uint8).reshape(img.size[1], img.size[0], 3)

    def crop_image_from_gray(self, img, tol=7):
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = skimage.color.rgb2grey(img)
            mask = gray_img > tol

            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if (check_shape == 0):  # image is too dark so that we crop out everything,
                return img  # return original image
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            return img

    def preprocessing(self, path, sigmaX=10):
        '''
        Preprocessing:
        Reading, croping and then converting to tensor and normalize to -1..1 with mean/std
        '''
        in_img = self.PIL2array(path)
        in_img = self.crop_image_from_gray(in_img)
        in_img = resize(in_img, (512, 512
                                ))
        in_img = img_as_ubyte(in_img) # convet to np.uint8

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                 std=[0.229, 0.224, 0.225])
        ])

        img_tensor = preprocess(in_img).unsqueeze(0).cpu()
        return img_tensor, in_img

    def getCAM(self, feature_conv, weight_fc, class_idx):
        '''

        :param feature_conv: final convnet layer in
        :param weight_fc: weights for predicted class
        :param class_idx: class index
        :return: heat map of roi

         We index into the fully-connected layer to get the
         weights for that class and calculate the dot product
         with our features from the image.
        '''
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv[0, :, :, ].reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return cam_img

    def make_predict(self, image):
        # Load predtrained Model
        model = resnet18
        ### Last layer's features for heatmap
        final_layer = model._modules.get('layer4')
        activated_features = SaveFeatures(final_layer)
        ####
        img_tensor, in_img = self.preprocessing(image)
        prediction = model(img_tensor)

        probabilities = F.softmax(prediction, dim=1).data.squeeze()
        label = np.argmax(probabilities.cpu().detach().numpy())
        #print(probabilities, '---', label)
        activated_features.remove()

        ## weights
        weight_softmax_params = list(model._modules.get('fc').parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

        ### Heat Map of ROI overlay
        overlay = self.getCAM(activated_features.features, weight_softmax, label)
        overlay = resize(overlay, (512, 512))
        cm_hot = mpl.cm.get_cmap('jet') # color map
        overlay = cm_hot(overlay)
        overlay = np.uint8(overlay * 255)
        overlay = Image.fromarray(overlay).convert("RGB")

        in_img = Image.fromarray(in_img)#.convert("RGB")

        # New image by interpolating between two images,
        # using a constant alpha.
        heatmap = Image.blend(in_img, overlay, 0.5)
        return probabilities, label, heatmap
