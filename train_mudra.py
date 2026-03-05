import torch
from torch.utils.data import DataLoader
from datasets.mudra_dataset import MudraDataset
from models.mudra_model import MudraNet

dataset = MudraDataset("data/mudra_images")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = MudraNet(len(dataset.classes))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        loss = loss_fn(model(imgs), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
