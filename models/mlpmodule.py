import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from quant import quantize


class MLPModule(pl.LightningModule):
    def __init__(self, input_size=3*32*32, hidden_sizes=[512, 256, 128], num_classes=100,
                 learning_rate=1e-3, dropout_rate=0.2, reg_scale=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.reg_scale = reg_scale
        
        # Build MLP layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # Track quantization state
        self.is_quantized = False
        self.quant_type = None
        self.quant_kwargs = None
    
    def apply_quantization(self, quant_type, quant_kwargs=None):
        """
        Apply quantization to the model after construction/loading.
        
        Args:
            quant_type: Type of quantization to apply
            quant_kwargs: Arguments for quantization
        """
        if quant_kwargs is None:
            quant_kwargs = {}
            
        if self.is_quantized:
            print(f"Model is already quantized with {self.quant_type}. Skipping.")
            return
            
        print(f"Applying {quant_type} quantization to model...")
        
        # Get linear layer names for quantization
        layer_names = self._get_quant_layer_names()
        self.mlp = quantize(self.mlp, layer_names=layer_names,
                           quant_type=quant_type, **quant_kwargs)
        
        # Update quantization state
        self.is_quantized = True
        self.quant_type = quant_type
        self.quant_kwargs = quant_kwargs
        
        print(f"Quantization applied successfully.")

    def _get_quant_layer_names(self):
        # Return names of all linear layers for quantization
        return [name for name, module in self.mlp.named_modules() 
                if isinstance(module, nn.Linear)]
    
    def forward(self, x):
        # Flatten the input if it's an image tensor
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.mlp(x)
    
    def reg_loss(self, reduction="mean"):
        """Compute regularization loss from quantized layers"""
        if not self.is_quantized:
            return torch.tensor(0.0, device=self.device)
            
        losses = []
        for m in self.mlp.modules():
            fn = getattr(m, "layer_reg_loss", None)
            if callable(fn):
                losses.append(fn())
        if not losses:
            return torch.tensor(0.0, device=self.device)
        losses = torch.stack([torch.as_tensor(l, device=self.device) for l in losses])
        return losses.mean() if reduction == "mean" else losses.sum()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        
        # Compute classification loss
        loss = F.cross_entropy(logits, labels)
        
        # Add regularization loss (only if quantized)
        rloss = self.reg_scale * self.reg_loss()
        total_loss = loss + rloss
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        log_dict = {
            "train_loss": loss,
            "total_loss": total_loss,
            "train_acc": acc
        }
        
        # Only log reg_loss if model is quantized
        if self.is_quantized:
            log_dict["reg_loss"] = rloss
            
        self.log_dict(log_dict, on_step=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        
        # Compute loss and accuracy
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log_dict({
            "val_loss": loss,
            "val_acc": acc
        }, prog_bar=True)
        
        return {"val_loss": loss, "val_acc": acc}
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        
        # Compute loss and accuracy
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log_dict({
            "test_loss": loss,
            "test_acc": acc
        })
        
        return {"test_loss": loss, "test_acc": acc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.mlp.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer


if __name__ == "__main__":
    # Example usage
    mlp = MLPModule(
        input_size=3*32*32,  # CIFAR-100 image size
        hidden_sizes=[512, 256, 128],
        num_classes=100,  # CIFAR-100 classes
        learning_rate=1e-3
    )
    
    # Test with dummy data
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 32, 32)
    dummy_labels = torch.randint(0, 100, (batch_size,))
    
    print("=== Full Precision Model ===")
    # Forward pass (full precision)
    logits = mlp(dummy_images)
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    print(f"Is quantized: {mlp.is_quantized}")
    
    # Apply quantization
    print("\n=== Applying Quantization ===")
    mlp.apply_quantization("tsvdlinear", {"rank": 8})
    
    # Test quantized model
    print("\n=== Quantized Model ===")
    logits_quant = mlp(dummy_images)
    print(f"Output shape after quantization: {logits_quant.shape}")
    print(f"Is quantized: {mlp.is_quantized}")
    print(f"Quantization type: {mlp.quant_type}")
    print(f"Quantization args: {mlp.quant_kwargs}")
    
    # Compare outputs
    diff = (logits - logits_quant).abs().mean()
    print(f"Mean absolute difference: {diff:.6f}")
    
    print("\n=== Usage Example for Checkpoint Loading ===")
    print("# 1. Train unquantized model and save checkpoint")
    print("# 2. Load checkpoint:")
    print("model = MLPModule.load_from_checkpoint('checkpoint.ckpt')")
    print("# 3. Apply quantization:")
    print("model.apply_quantization('tsvdlinear', {'rank': 8})")
    print("# 4. Continue training or inference with quantized model")
