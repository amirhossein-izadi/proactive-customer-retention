# PyTorch and Deep Learning

The job description says "Expertise with deep learning frameworks, particularly PyTorch." This section covers what you need to know to write, debug, and deploy PyTorch models.

## Why PyTorch

PyTorch is the dominant deep learning framework in research and is increasingly dominant in production (alongside TensorFlow and JAX). Its defining features:

- **Dynamic computation graphs**: the graph is built at runtime, which makes debugging and dynamic models (RNNs, recursive networks) natural.
- **Pythonic**: feels like NumPy with autograd.
- **Strong ecosystem**: torchvision, torchaudio, torchtext, transformers, accelerate, lightning, etc.
- **Production-ready**: TorchScript, ONNX export, torch.compile, broad deployment support.

The job description's preference for PyTorch reflects the industry norm.

## Core Concepts

### Tensors

The fundamental data structure. Multidimensional arrays with a single dtype.

```python
import torch

x = torch.tensor([1, 2, 3])              # 1D
x = torch.zeros(3, 4)                     # 2D
x = torch.randn(2, 3, 4)                 # 3D (batch, channels, length)

# Key attributes
x.shape        # tensor size
x.dtype        # data type (float32, int64, etc.)
x.device       # cpu or cuda
x.requires_grad  # whether gradients are tracked
```

Tensors can be moved between CPU and GPU with `.to(device)`. Common device patterns:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
data = data.to(device)
```

### Autograd

PyTorch's automatic differentiation engine. It tracks operations on tensors with `requires_grad=True` and computes gradients by reverse-mode autodiff (backpropagation).

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.])
```

Key points:
- `requires_grad=True` is sticky: operations produce tensors that also track gradients.
- `.backward()` computes gradients of a scalar with respect to leaf tensors.
- `torch.no_grad()` disables gradient tracking for inference.
- `model.eval()` switches to evaluation mode (disables dropout, uses running stats in batch norm).
- For non-scalar outputs, pass `gradient` to `.backward()`.

PyTorch's official tutorials describe `torch.autograd` as the automatic differentiation engine powering neural network training.

## Building Models with `nn.Module`

The standard way to define a model.

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
```

Key points:
- Subclass `nn.Module`.
- Define layers in `__init__`.
- Define the forward pass in `forward`.
- Use `nn.functional` for stateless operations.
- Layers with learnable parameters (Linear, Conv2d, BatchNorm1d) must be assigned to `self`.
- `nn.Module` automatically tracks all parameters via `model.parameters()`.

## Loss Functions

| Loss | Use case | Notes |
|---|---|---|
| `nn.CrossEntropyLoss` | multiclass classification | Input: raw logits. Internally applies softmax + NLL. |
| `nn.BCEWithLogitsLoss` | binary classification | Numerically stable. Input: raw logits, not sigmoid output. |
| `nn.BCELoss` | binary classification | Less stable than BCEWithLogits; requires sigmoid first. |
| `nn.MSELoss` | regression | Default reduction is mean. |
| `nn.L1Loss` | regression (robust) | Less sensitive to outliers. |
| `nn.HuberLoss` | regression (robust, smooth) | Combines L1 and L2; threshold-controlled. |
| `nn.CosineEmbeddingLoss` | similarity learning | |
| `nn.TripletMarginLoss` | metric learning | |
| `nn.CTCLoss` | sequence-to-sequence with variable length | |

Always use `BCEWithLogitsLoss` over `BCELoss` for numerical stability. Same for `CrossEntropyLoss` over manual softmax + NLL.

## Optimizers

```python
import torch.optim as optim

# Most common
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```

Key concepts:
- **SGD with momentum**: classic, often works well with proper LR schedule.
- **Adam**: adaptive learning rates per parameter. Fast convergence, good default.
- **AdamW**: Adam with decoupled weight decay. Generally preferred over Adam for regularization.
- **Learning rate**: the single most important hyperparameter.
- **Weight decay (L2 regularization)**: penalizes large weights.

### Learning rate schedules

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

Common patterns:
- Constant LR for short experiments.
- Step decay (multiply by 0.1 every N epochs).
- Cosine annealing (smooth decay).
- Warmup + decay (start low, ramp up, decay).

## The Standard Training Loop

This is what you must be able to write from memory in an interview.

```python
model = MLP(input_dim=784, hidden_dim=256, output_dim=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()                    # enable dropout, batch norm running stats
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()        # CRITICAL: clear previous gradients
        outputs = model(x)           # forward pass
        loss = criterion(outputs, y) # compute loss
        loss.backward()              # backpropagation
        optimizer.step()             # update weights

    # Validation
    model.eval()                     # disable dropout, use running stats
    with torch.no_grad():            # disable gradient tracking
        val_loss = 0
        correct = 0
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            val_loss += criterion(outputs, y).item() * x.size(0)
            correct += (outputs.argmax(1) == y).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, accuracy={accuracy:.4f}")
```

The five lines that actually do the work:
```python
optimizer.zero_grad()
outputs = model(x)
loss = criterion(outputs, y)
loss.backward()
optimizer.step()
```

Memorize them.

## DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
```

Key parameters:
- `batch_size`: trade off between speed (large) and noise (small).
- `shuffle`: True for training, False for validation/test.
- `num_workers`: parallel data loading. Set to 0 for debugging.
- `pin_memory`: faster transfer to GPU.
- `drop_last`: drop the last incomplete batch (helps with batch norm).

## Debugging Neural Networks

Common problems and fixes:

| Problem | Symptom | Fix |
|---|---|---|
| Loss is NaN | NaN values in loss/gradients | Lower LR, gradient clipping, check for log(0), check input scaling |
| Loss not decreasing | Flat loss curve | Check LR (too high or too low), check data labels, check model output |
| Loss decreasing, val loss flat | Overfitting | Regularization, more data, smaller model, early stopping |
| Loss and val loss both high | Underfitting | Larger model, more features, train longer |
| OOM (out of memory) | CUDA out of memory | Lower batch size, use mixed precision, use gradient accumulation |
| Slow training | Low GPU utilization | Larger batch size, more data loading workers, move data to GPU more efficiently |
| Predictions are constant | Model outputs same class | Check loss, check class balance, check model capacity |
| Gradients are zero | Dead ReLU or vanishing gradients | Change activation function, better initialization, residual connections |

### Gradient clipping

For RNNs and transformers, gradients can explode. Clip them:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Mixed precision training

For 2-3x speedup on modern GPUs:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in loader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(x)
        loss = criterion(outputs, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Or use `torch.cuda.amp.autocast` context manager.

## Checkpointing

Save and load model state.

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**Always save**:
- Model state dict.
- Optimizer state dict (to resume training).
- Current epoch.
- Best metric so far.
- Random seed state (for reproducibility).

For inference only, save just the state dict and the model definition.

## Device Management

Common pitfalls:

- Forgetting to move model to GPU.
- Forgetting to move data to GPU.
- Mixing CPU and GPU tensors in operations (error).
- Trying to convert numpy arrays on GPU to numpy (need `.cpu().numpy()`).
- Calling `.item()` on GPU tensors (works but forces sync).

## Common Architectures You Should Know

### MLP (Multi-Layer Perceptron)

The simplest deep model. Stack of fully connected layers with activations.

Use for: tabular data, small problems, baselines.

### CNN (Convolutional Neural Network)

For images, audio spectrograms, and other grid-like data.

Key concepts:
- **Convolution**: learns local patterns.
- **Pooling**: downsamples, adds translation invariance.
- **Receptive field**: the area of input that affects a particular output.
- **Strides and padding**: control output size.
- **Channels**: each filter detects a different feature.

Standard architectures: ResNet, EfficientNet, VGG (mostly historical).

### RNN / LSTM / GRU

For sequence data (text, time series).

Key concepts:
- **Hidden state**: passed between time steps.
- **Vanishing gradients**: standard RNNs struggle with long sequences.
- **LSTM/GRU**: gated variants that mitigate vanishing gradients.
- **Bidirectional**: process the sequence in both directions.

For most modern sequence tasks, transformers have replaced RNNs.

### Transformers

The dominant architecture for NLP and increasingly for vision, audio, and tabular.

Key concepts:
- **Self-attention**: each token attends to all other tokens, weighted by learned scores.
- **Multi-head attention**: multiple attention patterns in parallel.
- **Positional encoding**: gives the model information about token order.
- **Encoder vs decoder**: encoders understand, decoders generate.
- **BERT-style encoders**: classification, retrieval, embedding.
- **GPT-style decoders**: text generation.

Key formulas:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### Transfer Learning

Use a pretrained model and fine-tune on your task.

```python
import torchvision.models as models
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze backbone
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
```

Common patterns:
- Freeze backbone, train new head.
- Unfreeze last few layers, train with low LR.
- Full fine-tune with very low LR.

For NLP, fine-tuning a BERT-style model is the standard.

## Regularization for Deep Learning

| Technique | Effect |
|---|---|
| Dropout (0.1-0.5) | Randomly zero activations during training |
| Weight decay (1e-4 to 1e-2) | L2 penalty on weights |
| Data augmentation | More effective training data |
| Early stopping | Stop before overfitting |
| BatchNorm / LayerNorm | Implicit regularization, faster training |
| Label smoothing | Soft targets to prevent overconfidence |
| Mixup / CutMix | Combine samples and labels |

## The Debugging Workflow

When a model is not working, follow this order:

1. **Check the data**: visualize samples, check labels, check normalization.
2. **Overfit a small batch**: model should get to near-zero loss on 1-2 batches.
3. **Check loss curve**: should decrease, not flat or NaN.
4. **Check gradients**: should not be zero or exploding.
5. **Compare to baseline**: a linear model on the same data should at least be reasonable.
6. **Check the metric**: is the evaluation metric what you think it is?
7. **Slice analysis**: is the model failing on specific subgroups?

If step 2 fails, the model has a bug. Fix it before doing anything else.

## Common Interview Topics for PyTorch

1. **Write a training loop from scratch**: memorize the 5 lines.
2. **Explain backpropagation**: chain rule through computation graph.
3. **Debug NaN loss**: gradient explosion, learning rate too high, log(0).
4. **Why use `BCEWithLogitsLoss` instead of `BCELoss`?**: numerical stability.
5. **Difference between `model.eval()` and `torch.no_grad()`**: `eval()` changes layer behavior; `no_grad()` skips gradient tracking.
6. **Transfer learning**: how to fine-tune a pretrained model.
7. **GPU memory management**: batch size, mixed precision, gradient accumulation.
8. **How to handle class imbalance**: weighted loss, sampling, focal loss.
9. **Overfitting control**: dropout, weight decay, augmentation, early stopping.
10. **Production deployment**: ONNX export, TorchScript, torch.compile, model serving.

## The Senior Interview Answer

When asked "how do you train a deep learning model in PyTorch?", the answer should cover:

1. **Data**: clean, normalized, augmented, with proper train/val/test split.
2. **Model**: architecture choice with rationale; use pretrained if applicable.
3. **Loss**: appropriate for the task; use numerically stable versions.
4. **Optimizer**: AdamW with proper LR; LR schedule if training long.
5. **Training loop**: forward → loss → backward → step, with proper zero_grad.
6. **Validation**: held-out set; metric matches business objective.
7. **Debugging**: check data first, then overfit a small batch, then full training.
8. **Regularization**: dropout, weight decay, augmentation, early stopping.
9. **Checkpointing**: save best model by validation metric; resume from checkpoint.
10. **Deployment**: export (ONNX/TorchScript), latency test, monitoring after deploy.
