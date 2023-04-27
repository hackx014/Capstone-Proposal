import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # fix the input shape
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)  # fix the input shape
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_fn(model_dir):
    logger.info('Loading the model.')
    model = Net().to(device)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    logger.info('Model loaded successfully.')
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type == JSON_CONTENT_TYPE:
        request = json.loads(request_body)
        url = request['url']
        img_content = requests.get(url).content
        image = Image.open(io.BytesIO(img_content)).convert('RGB')
    elif content_type == JPEG_CONTENT_TYPE:
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
    else:
        raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))
    return image

def predict_fn(input_object, model):
    logger.info('Processing the input data.')
    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_object)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    with torch.no_grad():
        output = model(input_batch)
        output = output.cpu().numpy()
    return output.tolist()
