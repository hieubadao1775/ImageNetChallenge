from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.nn import Softmax
from model import MyAlex
import torch
import cv2
from dataset import Animal
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="test_image/dog.webp")

    args = parser.parse_args()
    return args

def test(args):
    data = Animal(root="../data/animals/train")
    categories = sorted(data.categories)

    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    softmax = Softmax(dim=1)

    checkpoint = torch.load("checkpoint/best.pt")
    model = MyAlex(10)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ori_image = cv2.imread(args.root)
    img = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    output= model(img)
    output = softmax(output)

    category = categories[torch.argmax(output).item()]
    confidence = output[0][torch.argmax(output)]

    ori_image = cv2.putText(ori_image, f"{category}-{confidence:.2f}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 255), thickness=2)
    cv2.imshow("Prediction", ori_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    args = get_args()
    test(args)