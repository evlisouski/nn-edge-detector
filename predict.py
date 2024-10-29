import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from models.model_2 import EdgeDetector


def load_model(model_path="deep_edge_model2.pth"):
    model = EdgeDetector()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def predict_edges(model, image_path, threshold=0):
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),])
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    input_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_image)
    output_image = output.squeeze().cpu().numpy()
    output_image = Image.fromarray(output_image)
    output_image = output_image.resize((width, height), Image.Resampling.BILINEAR)
    output_image = np.array(output_image)
    if threshold > 0:
        output_image = (output_image > threshold).astype(np.uint8)
    return output_image
