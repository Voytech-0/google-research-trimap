import shutil
from pyexpat import features

from torchvision import datasets, models, transforms
from torch import nn
import torch
from torchmetrics.classification import Accuracy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import numpy as np


def load_pretrained_backbone(backbone='resnet50'):
    if backbone == 'resnet50':
        return models.resnet50(pretrained=True)
    elif backbone == 'resnet18':
        return models.resnet18(pretrained=True)
    else:
        raise ValueError(f'{backbone} is an unknown backbone')

def load_model(num_classes=7, backbone='resnet50'):
    model = load_pretrained_backbone(backbone)

    for param in model.parameters(): # freeze backbone
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class LightningModel(pl.LightningModule):
    def __init__(self, num_classes=7, lr=1e-4, backbone='resnet50'):
        super().__init__()
        self.save_hyperparameters()
        self.model = load_model(num_classes=num_classes, backbone=backbone)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def train_model(train_dataloader, eval_dataloader, seed=0, max_epochs=2):
    pl.seed_everything(seed)

    model = LightningModel(num_classes=7)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1, verbose=True
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # uses GPU if available
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=eval_dataloader)

    best_path = checkpoint_callback.best_model_path
    if best_path:
        shutil.copy(best_path, "best.ckpt")
    return model

def get_embeddings(model, dataloader):
    features, predictions, actual = None, None, None
    model.eval()
    device = next(model.parameters()).device
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    classifier = list(model.children())[-1]
    for x, y in dataloader:
        x = x.to(device)
        embeddings = feature_extractor(x).squeeze()
        predicted = classifier(embeddings).argmax(dim=-1)

        if features is None or predictions is None or actual is None:
            features = embeddings.cpu().detach().numpy()
            predictions = predicted.cpu().detach().numpy()
            actual = y.cpu().detach().numpy()
        else:
            features = np.concatenate((features, embeddings.cpu().detach().numpy()))
            predictions = np.concatenate((predictions, predicted.cpu().detach().numpy()))
            actual = np.concatenate((actual, y.cpu().detach().numpy()))

    return features, predictions, actual


