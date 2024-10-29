import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.model_2 import EdgeDetector
from dataset_loaders.bsds500 import BDSD500Dataloader as DatasetsLoader


def train(list_with_pair="datasets/BSDS500/train_pair.lst", num_epochs=10, info_freq=50):    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # DataLoader
    train_dataset = DatasetsLoader(list_with_pair, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    # Initialization of the model, loss function and optimizer
    model = EdgeDetector()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training cycle
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0    
        for i, (inputs, labels) in enumerate(train_loader):        
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Back propagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % info_freq == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    torch.save(model.state_dict(), f'edge_detector_{EdgeDetector.__name__}_{epoch+1}.pth')

if __name__ == "__main__":
    list_with_pair = "datasets/BSDS500/train_pair.lst"  # Path to a list with image pairs for training
    num_epochs = 2  # Number of epochs for training
    info_freq = 50  # frequency of outputting information to the command line
    
    train(list_with_pair=list_with_pair, num_epochs=num_epochs, info_freq=info_freq)
