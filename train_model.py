import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np

def train(model, train_loader, val_loader, class_names, epochs, learning_rate, device):
    """
    Training function that takes a model and data loaders.
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0.01)
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
        
        scheduler.step(avg_val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.2%}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
            print(f"  → New best validation accuracy saved to best_model.pt: {best_val_acc:.2%}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {patience_limit} epochs without improvement")
            break
        
        # This fine-tuning logic is specific to ResNet and might need adjustment
        if epoch == 5:
            print("Unfreezing more layers for fine-tuning...")
            if hasattr(model, 'layer4'):
                for param in model.layer4.parameters():
                    param.requires_grad = True
                # Re-create optimizer to include newly unfrozen layers
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate/10, weight_decay=0.01)

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2%}")
    Path("labels.txt").write_text("\n".join(class_names))

def get_data_loaders(data_dir, batch_size, image_size=(224, 224)):
    """Prepares and returns train/validation dataloaders."""
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
    
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    print(f"Found {len(class_names)} classes: {class_names}")

    train_size = int(0.8 * len(full_dataset))
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(datasets.ImageFolder(data_dir, transform=train_transform), train_indices)
    val_dataset = Subset(datasets.ImageFolder(data_dir, transform=val_transform), val_indices)

    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, class_names

def train_from_scratch():
    """Sets up a new model and starts training from scratch."""
    # Parameters
    data_dir = "dataset"
    batch_size = 16 # Using smaller batch size for less intensive training
    epochs = 30     # Using fewer epochs
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size)
    
    model = models.resnet50(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )
    print("Setting up a new model for training from scratch.")
    # Call the training engine
    print(f'Using device: {device}')
    train(model, train_loader, val_loader, class_names, epochs, learning_rate, device)

def fine_tune():
    """Loads and expands a model, then starts the training process."""
    # Parameters
    data_dir = "dataset"
    model_path = "best_model.pt" # Load the best model, not the final one
    batch_size = 16
    epochs = 1
    learning_rate = 0.00005 # Use a smaller learning rate for fine-tuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size)
    num_classes = len(class_names)
    
    # 1. Instantiate the new, expanded model architecture
    new_model = models.resnet50() # Start with the base ResNet-50
    for param in new_model.parameters(): # Freeze all layers initially
        param.requires_grad = False

    new_model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=new_model.fc.in_features, out_features=512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(in_features=512, out_features=256), # New Layer
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=256, out_features=num_classes)
    )
    
    # 2. Load the state_dict from your saved model file
    try:
        # Load the state dict. strict=False allows loading weights for non-matching architectures.
        pretrained_dict = torch.load(model_path, map_location=device)
        new_model.load_state_dict(pretrained_dict, strict=False)
        print(f"Successfully loaded pre-trained weights from {model_path}")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return

    # Unfreeze the parameters of the new classifier so they can be trained
    for param in new_model.fc.parameters():
        param.requires_grad = True

    # 3. Call the main train function with the prepared model and data
    print("Starting fine-tuning on the expanded model.")
    print(f'Using device: {device}')
    train(new_model, train_loader, val_loader, class_names, epochs, learning_rate, device)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    choice = input("Select mode:\n1. Train from scratch\n2. Fine-tune existing model\nEnter choice (1/2): ")
    if choice == "1":
        train_from_scratch()
    elif choice == "2":
        fine_tune()