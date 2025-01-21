import torch
from torch import nn
from torchvision.transforms import Normalize
import cv2



mapping_dict = {}
with open('emnist-balanced-mapping.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split()
        mapping_dict[int(key)] = int(value)

class Cnn_model(nn.Module):
    def __init__(self, inp: int, hidden_u: int, output: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(inp, hidden_u, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_u),
            nn.Conv2d(hidden_u, hidden_u, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_u),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*hidden_u * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(256, output)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.classifier(x)
        return x


normalizer = Normalize([0.5], [0.5])


class Model():
    def __init__(self):
        self.model = Cnn_model(inp = 1,
                            hidden_u = 20,
                            output = len(mapping_dict))
        self.model.load_state_dict(torch.load('myapp/model_3.ckpt')
                                   )
        self.model.eval()

    def predict(self, x):
#        x = 255 - x.numpy()
        x = cv2.GaussianBlur(x.numpy(), (3, 3), 0)
        x = cv2.dilate(x, (3, 3)).reshape(1, 1, 28, 28)
        x = torch.Tensor(x)

        x_normalized = normalizer(x.float())

        output = self.model(x_normalized)
        prediction = int(torch.argmax(output, dim=1))
        return chr(mapping_dict[prediction])
