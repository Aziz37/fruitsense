import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
train_images = np.load('data\\Avocado\\X_full8_train.npy')
train_labels = np.load('data\\Avocado\\y_classify_full8_train.npy')
train_labels = [int(x) - 1 for x in train_labels]

val_images = np.load('data\\Avocado\\X_full8_val.npy')
val_labels = np.load('data\\Avocado\\y_classify_full8_val.npy')
val_labels = [int(x) - 1 for x in val_labels]

test_images = np.load('data\\Avocado\\X_full8_test.npy')
test_labels = np.load('data\\Avocado\\y_classify_full8_test.npy')
test_labels = [int(x) - 1 for x in test_labels]

# Compute mean and std per channel over the training data
mean = train_images.mean(axis=(0, 1, 2))  # Shape: (8,)
std = train_images.std(axis=(0, 1, 2))    # Shape: (8,)

# Normalize the datasets using the computed mean and std
train_images = (train_images - mean) / std
val_images = (val_images - mean) / std
test_images = (test_images - mean) / std

# Compute class counts for the training labels
class_counts = np.bincount(train_labels)
num_classes = len(class_counts)
total_samples = len(train_labels)

# Compute class weights: inversely proportional to class frequencies
class_weights = total_samples / (num_classes * class_counts)
print(f"Class Weights: {class_weights}")

# Convert class weights to a PyTorch tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Convert images and labels to tensors
train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.int64)

# Create the training dataset and dataloader with adjusted batch size
batch_size = 64  # Adjusted batch size from 32 to 64
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Also adjust the validation data
val_images = torch.tensor(val_images, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.int64)
val_dataset = TensorDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Enhanced model definition remains the same
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Second convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Third convolutional block
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)

        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modify FocalLoss to accept class weights and num_classes
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', num_classes=num_classes):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        if alpha is None:
            self.alpha = torch.ones(self.num_classes, dtype=torch.float32)
        else:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha] * self.num_classes, dtype=torch.float32)
            elif isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                raise TypeError('alpha must be float, int, list, np.ndarray, or torch.Tensor')

    def forward(self, inputs, targets):
        # Ensure alpha is on the same device as inputs
        alpha = self.alpha.to(inputs.device)
        # Convert inputs to probabilities using softmax
        probs = F.softmax(inputs, dim=1)
        probs = probs.clamp(min=1e-8, max=1 - 1e-8)  # Avoid zero probabilities
        # Get the probabilities corresponding to the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        # Compute the loss
        log_pt = torch.log(pt)
        alpha_class = alpha[targets]
        loss = -alpha_class * ((1 - pt) ** self.gamma) * log_pt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the device
model = EnhancedCNN().to(device)

# Move class weights tensor to the same device as model
class_weights_tensor = class_weights_tensor.to(device)

# Define the loss function and optimizer with adjusted learning rate
learning_rate = 0.0005  # Adjusted learning rate from 0.001 to 0.0005
focal_loss = FocalLoss(alpha=class_weights_tensor, gamma=2, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Include a learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training loop with validation
num_epochs = 500
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    for inputs, labels in train_loader:
        # Permute inputs to match [batch_size, channels, height, width]
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = focal_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation step
    model.eval()
    running_val_loss = 0.0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.permute(0, 3, 1, 2).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = focal_loss(outputs, labels)
            running_val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().tolist())
            val_targets.extend(labels.cpu().tolist())

    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = accuracy_score(val_targets, val_preds)
    val_accuracies.append(val_accuracy)

    # Step the scheduler
    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plotting the validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.show()

# Evaluate on the test set
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.int64)

test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_preds = []
test_targets = []
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().tolist())
        test_targets.extend(labels.cpu().tolist())

test_accuracy = accuracy_score(test_targets, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate and plot the confusion matrix
cm = confusion_matrix(test_targets, test_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()