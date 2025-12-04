import yaml
import torch
from utils.data_loader import get_dataloaders
from utils.models import get_model
from utils.metrics import evaluate_model
import os
import json

config = yaml.safe_load(open("config.yaml"))
os.makedirs("results", exist_ok=True)

def train(model_name):
    train_loader, val_loader = get_dataloaders(config)
    model = get_model(model_name)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = 0

    for epoch in range(config["training"]["epochs"]):
        model.train()

        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        auc = evaluate_model(model, val_loader, model_name)
        print(f"{model_name} Epoch {epoch+1}: AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"results/{model_name}_best.pth")

    return best_auc

if __name__ == "__main__":
    results = {}
    for model_name in config["models"]:
        results[model_name] = train(model_name)

    json.dump(results, open("results/metrics.json", "w"), indent=4)
