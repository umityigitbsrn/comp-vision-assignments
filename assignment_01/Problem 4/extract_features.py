import scipy.io as sio
import torch.cuda
import torchvision
from models.alexnet import alexnet
from models.vgg import vgg16
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    alex_feature = []
    alex_label = []

    vgg16_feature = []
    vgg16_label = []

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)

    batch_size = 500
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # [Problem 4 a.] IMPORT VGG16 AND ALEXNET FROM THE MODELS FOLDER WITH
    PRETRAINED = True

    vgg16_extractor = vgg16(PRETRAINED).to(device)
    vgg16_extractor.eval()

    alex_extractor = alexnet(PRETRAINED).to(device)
    alex_extractor.eval()
    with torch.no_grad():
        for idx, data in enumerate(train_loader):
            image, label = data
            image, label = image.to(device), label.to(device)

            # [Problem 4 a.] OUTPUT VARIABLE F_vgg and F_alex EXPECTED TO BE THE
            # FEATURE OF THE IMAGE OF DIMENSION (4096,) AND (256,), RESPECTIVELY.
            F_vgg = vgg16_extractor(image)

            if (idx + 1) % 50 == 0 or idx == 0:
                print('iter: {}, F_vgg size: {}'.format(idx + 1, F_vgg.size()))

            vgg16_feature.append(F_vgg.to('cpu'))
            vgg16_label.append(label.to('cpu'))

            F_alex = alex_extractor(image)

            if (idx + 1) % 50 == 0 or idx == 0:
                print('iter: {}, F_alex size: {}'.format(idx + 1, F_alex.size()))

            alex_feature.append(F_alex.to('cpu'))
            alex_label.append(label.to('cpu'))

            if (idx + 1) % 50 == 0 or idx == 0:
                print('iter: {}, len alex_feature: {}, vgg_feature: {}'.format(idx + 1, len(alex_feature),
                                                                               len(vgg16_feature)))

    vgg16_feature = torch.cat(vgg16_feature, 0)
    vgg16_label = torch.cat(vgg16_label, 0)

    alex_feature = torch.cat(alex_feature, 0)
    alex_label = torch.cat(alex_label, 0)

    sio.savemat('vgg16.mat', mdict={'feature': vgg16_feature.numpy(), 'label': vgg16_label.numpy()})
    sio.savemat('alexnet.mat', mdict={'feature': alex_feature.numpy(), 'label': alex_label.numpy()})


# def KNN_test(train_mat_file, test_data, K=1):
#     # FILL IN TO LOAD THE SAVED .MAT FILE
#     vgg_mat =
#     alex_mat =
#
#     vgg16_extractor =
#     alex_extractor =
#
#     for idx, data in enumerate(test_data):
#         # 1. # EXTRACT FEATURES USING THE MODELS - ALEXNET AND VGG16
#         F_test_vgg16 =
#         F_test_alex =
#
#         # 2. # FIND NEAREST NEIGHBOUR OF THIS FEATURE FROM FEATURES STORED IN ALEXNET.MAT AND VGG16.MAT
#
#         # 3. # COMPUTE ACCURACY
#         alex_accuracy = 0.0
#         vgg16_accuracy = 0.0
#
#         return vgg16_accuracy, alex_accuracy


if __name__ == "__main__":
    main()
