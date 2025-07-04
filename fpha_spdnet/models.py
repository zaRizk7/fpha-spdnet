from lightning import LightningModule
from spdnet import SPDNet as SPDNetBackbone
from spdnet import USPDNet as USPDNetBackbone
from spdnet.metrics import distance
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from torch.nn.functional import cross_entropy as ce
from torchmetrics import Accuracy

__all__ = ["SPDNet", "USPDNet"]


class SPDNet(LightningModule):
    """
    PyTorch Lightning module for classification of SPD matrices using SPDNet.

    Args:
        num_spatials (list[int]): Output spatial dimensions for each SPD layer.
        num_classes (int): Number of target classes (1 for binary).
        matrix_size (int): Size of input SPD matrices (inferred from dataset).
        rectify_last (bool): Whether to apply ReLU after final SPD layer.
        use_batch_norm (bool): Whether to apply batch normalization after SPD layers.
        eps (float): Small constant for clamping eigenvalues with ReEig.
    """

    def __init__(
        self,
        num_spatials: list[int],
        num_classes: int,
        matrix_size: int,
        rectify_last: bool = False,
        use_batch_norm: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SPDNetBackbone(
            num_spatials=[matrix_size] + num_spatials,
            num_outputs=num_classes,
            rectify_last=rectify_last,
            use_batch_norm=use_batch_norm,
            eps=eps,
        )

        is_binary = num_classes == 1
        self.loss_fn = bce if is_binary else ce
        task = "binary" if is_binary else "multiclass"
        self.train_accuracy = Accuracy(task, num_classes=num_classes)
        self.val_accuracy = self.train_accuracy.clone()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        self.train_accuracy(y_hat, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        self.val_accuracy(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)

        return loss


class USPDNet(LightningModule):
    """
    PyTorch Lightning module for classification of SPD matrices using USPDNet.

    Args:
        num_spatials (list[int]): Output spatial dimensions for each SPD layer.
        num_classes (int): Number of target classes (1 for binary).
        matrix_size (int): Size of input SPD matrices.
        use_batch_norm (bool): Whether to apply batch normalization after SPD layers.
        eps (float): Small constant for clamping eigenvalues with ReEig.
        trade_off (float): Trade-off coefficient for reconstruction loss.
    """

    def __init__(
        self,
        num_spatials: list[int],
        num_classes: int,
        matrix_size: int,
        use_batch_norm: bool = False,
        eps: float = 1e-5,
        trade_off: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = USPDNetBackbone(
            num_spatials=[matrix_size] + num_spatials,
            num_outputs=num_classes,
            use_batch_norm=use_batch_norm,
            eps=eps,
        )

        self.trade_off = trade_off

        is_binary = num_classes == 1
        self.loss_fn = bce if is_binary else ce
        task = "binary" if is_binary else "multiclass"
        self.train_accuracy = Accuracy(task, num_classes=num_classes)
        self.val_accuracy = self.train_accuracy.clone()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat, y_hat = self.model(x)

        clf_loss = self.loss_fn(y_hat, y)
        rec_loss = (distance(x_hat, x, metric="euc") ** 2).mean()
        loss = clf_loss + self.trade_off * rec_loss
        self.train_accuracy(y_hat, y)

        self.log("train_clf_loss", clf_loss, on_epoch=True, prog_bar=True)
        self.log("train_rec_loss", rec_loss, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat, y_hat = self.model(x)

        clf_loss = self.loss_fn(y_hat, y)
        rec_loss = (distance(x_hat, x, metric="euc") ** 2).mean()
        loss = clf_loss + self.trade_off * rec_loss
        self.val_accuracy(y_hat, y)

        self.log("val_clf_loss", clf_loss, on_epoch=True, prog_bar=True)
        self.log("val_rec_loss", rec_loss, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)

        return loss
