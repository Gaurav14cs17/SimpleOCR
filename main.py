import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import os
import warnings
from dataloader import *

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class OCRModel(nn.Module):
    def __init__(self, num_chars, hidden_size=256, lstm_layers=2):
        super(OCRModel, self).__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 4 - Adjusted to output 256 channels
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Special pooling to maintain width for LSTM
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        # LSTM Sequence Model - input size must match CNN output features
        self.lstm = nn.LSTM(
            input_size=512,  # This MUST match channels * height from CNN output
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)  # +1 for CTC blank

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input shape: [batch, 1, height, width]

        # CNN features
        x = self.cnn(x)  # Output shape: [batch, 256, 1, width/8]

        # Prepare for LSTM
        batch_size, channels, height, width = x.size()

        # Reshape to: [sequence_length, batch, features]
        # sequence_length = width dimension
        # features = channels * height = 256 * 1 = 256
        x = x.permute(3, 0, 1, 2)  # [width, batch, channels, height]
        x = x.reshape(width, batch_size, -1)  # Flatten channels and height

        # Verify dimension matches LSTM input_size
        #assert x.size(-1) == 256, f"Expected 256 features, got {x.size(-1)}"

        # LSTM
        x, _ = self.lstm(x)

        # Output layer
        x = self.fc(x)
        return x

# # Custom collate function
# def collate_fn(batch):
#     """Handle variable-length sequences in batches"""
#     images = torch.stack([item['image'] for item in batch])
#     coords = torch.stack([item['coords'] for item in batch])
#
#     # Handle variable-length text sequences
#     texts = [item['text'] for item in batch]
#     text_ints = [item['text_int'] for item in batch]
#
#     # Get lengths of each sequence
#     text_lens = torch.tensor([len(t) for t in text_ints], dtype=torch.long)
#
#     # Pad sequences to maximum length in batch
#     max_len = max(text_lens) if len(text_lens) > 0 else 0
#     padded_text = torch.zeros(len(batch), max_len, dtype=torch.long)
#     for i, text in enumerate(text_ints):
#         if len(text) > 0:
#             padded_text[i, :len(text)] = text
#
#     return {
#         'image': images,
#         'text': texts,
#         'text_int': padded_text,
#         'text_len': text_lens,
#         'coords': coords
#     }


# Training Function
def train_model(dataset, char_to_int, device, num_epochs=20, batch_size=16):
    # Initialize model
    num_chars = len(char_to_int)
    model = OCRModel(num_chars).to(device)

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=num_chars, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        processed_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['text_int'].to(device, non_blocking=True)
            target_lengths = batch['text_len'].to(device, non_blocking=True)

            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)

            # Calculate input lengths (time steps)
            input_lengths = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)

            # Verify lengths are valid
            #print(target_lengths ,  input_lengths)
            if (target_lengths > input_lengths).any():
                print("Skipping batch with invalid length")
                continue

            # Calculate loss with numerical stability checks
            log_probs = outputs.log_softmax(2).clamp(min=-100)  # Prevent -inf

            try:
                loss = criterion(
                    log_probs,  # (T, N, C)
                    targets,  # (N, S)
                    input_lengths,  # (N)
                    target_lengths  # (N)
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    print("Invalid loss value, skipping batch")
                    continue

            except RuntimeError as e:
                print(f"Error calculating loss: {e}")
                continue

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            # Calculate accuracy
            with torch.no_grad():
                _, max_indices = torch.max(outputs, 2)
                max_indices = max_indices.permute(1, 0)  # (N, T)

                for i in range(len(targets)):
                    pred_text = []
                    prev_char = None
                    for idx in max_indices[i]:
                        if idx != num_chars and idx != prev_char:
                            pred_text.append(idx.item())
                        prev_char = idx

                    true_text = targets[i][:target_lengths[i]].tolist()
                    if pred_text == true_text:
                        correct += 1
                    total += 1

            epoch_loss += loss.item()
            processed_batches += 1

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {correct / max(total, 1):.4f}")

        if processed_batches == 0:
            print(f"Epoch {epoch + 1} had no valid batches")
            continue

        avg_loss = epoch_loss / processed_batches
        accuracy = correct / max(total, 1)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed, "
              f"Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Step the scheduler
        scheduler.step(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'char_to_int': char_to_int
            }, 'best_ocr_model.pth')
            print("Saved best model")

    return model


# Main Execution
if __name__ == "__main__":
    # Character mappings
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    char_to_int = {c: i for i, c in enumerate(chars)}

    # Initialize dataset (make sure OCRDataset is properly defined)
    dataset = OCRDataset(
        annotation_dir='D:/synlabs/contrain_ocr/Data/train_images/labels',
        image_dir='D:/synlabs/contrain_ocr/Data/train_images/images',
        char_to_int=char_to_int,
        img_size=(100, 32)
    )

    print(f"Dataset contains {len(dataset)} samples")

    # Train model
    trained_model = train_model(
        dataset=dataset,
        char_to_int=char_to_int,
        device=device,
        num_epochs=20,
        batch_size=16
    )
