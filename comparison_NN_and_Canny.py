import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from predict import load_model, predict_edges
from sklearn.metrics import precision_score, recall_score, f1_score


def canny_detect_edges(image_path, sigma=1.0):
    original_image = Image.open(image_path).convert("RGB")
    image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
    edges_canny = cv2.Canny(blurred_image, 100, 200)
    return edges_canny


def read_original_image(image_path):
    original_image_pil = Image.open(image_path).convert("RGB")
    original_image_cv = cv2.cvtColor(np.array(original_image_pil), cv2.COLOR_RGB2BGR)
    return original_image_cv

def show_results(original_image_cv, output_image, edges_canny):
    '''Displays the original image, the edges predicted by the model,
    and the bounedges daries obtained through the Canny filter.'''

    plt.figure(figsize=(18, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB))  # Конвертация обратно в RGB для отображения
    plt.axis("off")

    # NN image
    plt.subplot(1, 3, 2)
    plt.title("NN EdgeDetector")
    plt.imshow(output_image, cmap="gray")
    plt.axis("off")

    # Canny image
    plt.subplot(1, 3, 3)
    plt.title("Canny")
    plt.imshow(edges_canny, cmap="gray")
    plt.axis("off")

    plt.show()


def calculate_score(images_path='datasets/BSDS500/train/aug_data/0.0_1_0',
                    masks_path='datasets/BSDS500/train/aug_gt/0.0_1_0',
                    model_path="saved_models/edge_detector_model_2.pth"):

    model = load_model(model_path)

    image_paths = [os.path.join(images_path, fname) for fname in os.listdir(images_path)[:200]]
    mask_paths = [os.path.join(masks_path, fname) for fname in os.listdir(masks_path)[:200]]

    predicted_edges_nn = []
    predicted_edges_canny = []

    for img_path, mask_path in zip(image_paths, mask_paths):    
        nn_res = predict_edges(model=model, image_path=img_path)
        nn_res = (nn_res > 0.06).astype(np.uint8)
        predicted_edges_nn.append(nn_res)
                
        canny_res = canny_detect_edges(image_path=img_path)
        predicted_edges_canny.append(canny_res)

    true_masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths]

    def calculate_metrics(predicted_edges, true_masks):
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        num_images = len(predicted_edges)

        for pred_edge, true_mask in zip(predicted_edges, true_masks):
            predicted_edges_binary = (pred_edge > 0).astype(int)
            true_masks_binary = (true_mask > 0).astype(int)

            precision = precision_score(true_masks_binary.flatten(), predicted_edges_binary.flatten())
            recall = recall_score(true_masks_binary.flatten(), predicted_edges_binary.flatten())
            f1 = f1_score(true_masks_binary.flatten(), predicted_edges_binary.flatten())

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        average_precision = total_precision / num_images
        average_recall = total_recall / num_images
        average_f1 = total_f1 / num_images

        return average_precision, average_recall, average_f1

    # Вычисление метрик для нейронной сети
    precision_nn, recall_nn, f1_nn = calculate_metrics(predicted_edges_nn, true_masks)
    # Вычисление метрик для Canny
    precision_canny, recall_canny, f1_canny = calculate_metrics(predicted_edges_canny, true_masks)
    # Вывод результатов
    print(f"NN Precision: {precision_nn:.4f}, Recall: {recall_nn:.4f}, F1: {f1_nn:.4f}")
    print(f"Canny Precision: {precision_canny:.4f}, Recall: {recall_canny:.4f}, F1: {f1_canny:.4f}")


if __name__ == "__main__":
    import random


    model_path = "saved_models/edge_detector_model_2.pth"  
    images_path = "datasets/BSDS500/train/aug_data/0.0_1_0"
    masks_path = "datasets/BSDS500/train/aug_gt/0.0_1_0"  
    random_image_from_dataset = random.choice([os.path.join(images_path, fname) for fname in os.listdir(images_path)[:200]])
    
    model = load_model(model_path)    
    edges_predict = predict_edges(model, image_path=random_image_from_dataset)
    edges_canny = canny_detect_edges(image_path=random_image_from_dataset)
    original_image = read_original_image(image_path=random_image_from_dataset)
    show_results(original_image, edges_predict, edges_canny)
    # calculate_score(images_path, masks_path, model_path)
