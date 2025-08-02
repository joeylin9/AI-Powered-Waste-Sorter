import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader, Subset
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np

def train():
    # Parameters
    data_dir = "dataset"
    batch_size = 32
    image_size = (224, 224)
    epochs = 40
    learning_rate = 0.0001

    # Enhanced data transforms with normalization
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=360),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset with proper transform handling
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Total samples: {len(full_dataset)}")

    # Better train/val split that preserves class distribution
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Create indices for stratified split (approximate)
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create datasets with proper transforms
    train_dataset = Subset(datasets.ImageFolder(data_dir, transform=train_transform), train_indices)
    val_dataset = Subset(datasets.ImageFolder(data_dir, transform=val_transform), val_indices)

    # Data loaders - optimized for GPU if available
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = models.resnet50(weights='IMAGENET1K_V1')

    # Freeze early layers for better transfer learning
    for param in model.parameters():
        param.requires_grad = False

    # Only train the final layers initially
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )

    model.to(device)

    # Better optimizer and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training tracking
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 7

    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.2%}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, "test_model.pt")
            patience_counter = 0
            print(f"  → New best validation accuracy: {best_val_acc:.2%}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {patience_limit} epochs without improvement")
            break
        
        # Unfreeze more layers after a few epochs for fine-tuning
        if epoch == 5:
            print("Unfreezing more layers for fine-tuning...")
            for param in model.layer4.parameters():  # Unfreeze last ResNet block
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate/10, weight_decay=0.01)

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2%}")

    # Save final model and labels
    torch.save(model.state_dict(), "test_model.pt")
    Path("labels.txt").write_text("\n".join(class_names))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train()