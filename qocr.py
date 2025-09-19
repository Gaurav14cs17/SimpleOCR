import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import warnings
from torch.nn import functional as F

warnings.filterwarnings('ignore')

# Character set for OCR
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_INT = {c: i + 1 for i, c in enumerate(CHARS)}  # +1 because 0 is for CTC blank
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank token


def four_point_transform(image: np.ndarray, coords: List[int],
                         output_size: Tuple[int, int] = (100, 32)) -> np.ndarray:
    """Crop using 4 coords [x1,y1,x2,y2] and resize to (W,H)."""
    x1, y1, x2, y2 = coords[:4]
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        # Return a blank image if coordinates are invalid
        return np.zeros(output_size[::-1], dtype=np.uint8)

    warped = image[y1:y2, x1:x2]
    warped = cv2.resize(warped, output_size)
    return warped


class OCRDataset(Dataset):
    def __init__(self, annotation_dir: str, image_dir: str, char_to_int: Dict[str, int],
                 img_size: Tuple[int, int] = (100, 32), transform=None):
        """
        OCR Dataset for annotation format: x1,y1,x2,y2,text
        """
        self.annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.char_to_int = char_to_int
        self.img_width, self.img_height = img_size
        self.transform = transform

        self.annotations = []
        self.annotation_map = {}

        for file_idx, file_name in enumerate(self.annotation_files):
            file_path = os.path.join(self.annotation_dir, file_name)
            file_annotations = self._load_annotation(file_path)
            for ann_idx, annotation in enumerate(file_annotations):
                unique_id = len(self.annotations)
                self.annotations.append(annotation)
                self.annotation_map[unique_id] = (file_idx, ann_idx)

    def _load_annotation(self, file_path: str) -> List[Dict]:
        """Load annotation file in format: x1,y1,x2,y2,text"""
        annotations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    try:
                        x1, y1, x2, y2 = map(int, parts[:4])
                        text = ','.join(parts[4:]).strip()
                        if text:
                            annotations.append({
                                'coords': [x1, y1, x2, y2],
                                'text': text
                            })
                    except ValueError:
                        continue
        return annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        annotation = self.annotations[idx]
        file_idx, _ = self.annotation_map[idx]
        file_name = self.annotation_files[file_idx]
        base_name = os.path.splitext(file_name)[0]

        # Load image
        img_path = os.path.join(self.image_dir, f"{base_name}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            # Try other extensions
            for ext in ['.png', '.jpeg', '.bmp']:
                img_path = os.path.join(self.image_dir, f"{base_name}{ext}")
                image = cv2.imread(img_path)
                if image is not None:
                    break
            if image is None:
                raise FileNotFoundError(f"Image not found: {base_name}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Crop + resize
        warped = four_point_transform(image, annotation['coords'], (self.img_width, self.img_height))

        # Normalize
        warped = warped.astype(np.float32) / 255.0
        warped = np.expand_dims(warped, axis=0)  # [1,H,W]

        # Encode text
        text = annotation['text'].upper()
        text_int = [self.char_to_int[c] for c in text if c in self.char_to_int]

        if len(text_int) == 0:
            # Skip empty text or add placeholder
            text_int = [1]  # Use '0' as placeholder

        if self.transform:
            warped = self.transform(warped)

        return {
            'image': torch.from_numpy(warped),
            'text': text,
            'text_int': torch.tensor(text_int, dtype=torch.long),
            'text_len': len(text_int),
            'coords': torch.tensor(annotation['coords'][:4], dtype=torch.float),
            'file_name': base_name
        }


def collate_fn(batch):
    """Custom collate function for OCR (handles variable-length labels)."""
    images = torch.stack([item['image'] for item in batch])
    coords = torch.stack([item['coords'] for item in batch])

    texts = [item['text'] for item in batch]
    text_ints = [item['text_int'] for item in batch]
    text_lens = torch.tensor([item['text_len'] for item in batch], dtype=torch.long)

    # Pad labels to max length
    max_len = max(text_lens).item() if len(text_lens) > 0 else 0
    padded_text = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(text_ints):
        if len(t) > 0:
            padded_text[i, :len(t)] = t

    return {
        'images': images,  # [B,1,H,W]
        'texts': texts,  # raw strings
        'text_ints': padded_text,  # [B, max_len] padded
        'text_lens': text_lens,  # [B] original lengths
        'coords': coords
    }


class QuantizedCRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class, map_to_seq_hidden=64, rnn_hidden=256,
                 leaky_relu=False):
        super(QuantizedCRNN, self).__init__()

        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Original CRNN components
        self.cnn, (output_channel, output_height, output_width) = self._cnn_backbone(img_channel, img_height, img_width,
                                                                                     leaky_relu)

        # Use regular LSTM instead of quantizable.LSTM for better compatibility
        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)
        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True, batch_first=False)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True, batch_first=False)
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0
        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]
        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            input_channel = channels[i]
            output_channel = channels[i + 1]

            cnn.add_module(
                f'conv{i}', nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        conv_relu(2)
        conv_relu(3)
        cnn.add_module('pooling2', nn.MaxPool2d(kernel_size=(2, 1)))
        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module('pooling3', nn.MaxPool2d(kernel_size=(2, 1)))
        conv_relu(6)

        output_channel, output_height, output_width = channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # Quantize input
        images = self.quant(images)

        # Original forward pass
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()
        conv = conv.reshape(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)

        seq = self.map_to_seq(conv)
        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)
        output = self.dense(recurrent)

        # Dequantize output
        output = self.dequant(output)
        return output

    def fuse_model(self):
        """Fuse Conv+BN+ReLU modules for quantization"""
        # Fuse Conv+BN+ReLU in the CNN backbone
        for name, module in self.cnn.named_children():
            if isinstance(module, nn.Sequential):
                # Check if this sequential block has Conv, BN, ReLU pattern
                if len(module) >= 3:
                    if (isinstance(module[0], nn.Conv2d) and
                            isinstance(module[1], nn.BatchNorm2d) and
                            isinstance(module[2], (nn.ReLU, nn.LeakyReLU))):
                        torch.ao.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True)

                # Also check for Conv+ReLU patterns (without BN)
                elif len(module) >= 2:
                    if (isinstance(module[0], nn.Conv2d) and
                            isinstance(module[1], (nn.ReLU, nn.LeakyReLU))):
                        torch.ao.quantization.fuse_modules(module, ['0', '1'], inplace=True)


def ctc_greedy_decode(output: torch.Tensor, chars: str = CHARS) -> List[str]:
    """Greedy decoder for CTC output"""
    _, max_indices = torch.max(output, dim=2)
    batch_size = output.size(1)
    decoded_texts = []

    for i in range(batch_size):
        indices = max_indices[:, i]
        # Remove consecutive duplicates and blank tokens (0)
        prev_index = -1
        decoded_chars = []
        for idx in indices:
            if idx != prev_index and idx != 0:
                # Convert index to character (index 1 -> char 0, index 2 -> char 1, etc.)
                if idx - 1 < len(chars):
                    decoded_chars.append(chars[idx - 1])
            prev_index = idx
        decoded_texts.append(''.join(decoded_chars))

    return decoded_texts


def train_qat_model(
        model: nn.Module,
        train_loader: DataLoader,
        weight_path: str = "",
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        device: str = 'cpu'
) -> nn.Module:
    """Train the model with Quantization-Aware Training"""
    if weight_path != "":
        try:
            state_dict = torch.load(weight_path, map_location=device)
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Filter out incompatible keys
            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if
                                   k in model_state_dict and v.shape == model_state_dict[k].shape}

            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded weights from {weight_path}, {len(filtered_state_dict)}/{len(state_dict)} parameters matched")
        except Exception as e:
            print(f"Error loading weights: {e}")

    model.train()
    model.to(device)

    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            targets = batch['text_ints'].to(device)
            target_lengths = batch['text_lens'].to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(images)

            # CTC Loss requires specific input format
            input_lengths = torch.full(
                size=(output.size(1),),
                fill_value=output.size(0),
                dtype=torch.long,
                device=device
            )

            # CTC loss expects: log_probs, targets, input_lengths, target_lengths
            log_probs = F.log_softmax(output, dim=2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}')

        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f'Epoch {epoch + 1}/{num_epochs} | Average Loss: {avg_loss:.4f}')

    return model


def prepare_and_convert_model(
        model: nn.Module,
        example_input: torch.Tensor,
        backend: str = 'fbgemm'
) -> nn.Module:
    """Prepare model for QAT and convert to INT8"""
    # Set backend
    torch.backends.quantized.engine = backend

    # Fuse modules
    model.fuse_model()

    # Set quantization configuration
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)

    # Prepare for QAT
    model_prepared = torch.ao.quantization.prepare_qat(model)

    print("Model prepared for QAT")
    return model_prepared


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu'):
    """Evaluate the model performance"""
    model.eval()
    model.to(device)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            targets = batch['texts']

            output = model(images)
            predictions = ctc_greedy_decode(output)

            # Compare predictions with targets
            for pred, true in zip(predictions, targets):
                if pred == true:
                    total_correct += 1
                total_samples += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f'Evaluation Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})')
    return accuracy


def compare_model_sizes(fp32_model: nn.Module, quantized_model_path: str):
    """Compare sizes of FP32 and INT8 models"""
    print("\nModel Size Comparison:")

    # Calculate FP32 model size
    fp32_size = sum(p.numel() * 4 for p in fp32_model.parameters()) / 1024 / 1024
    print(f"FP32 Model size: {fp32_size:.2f} MB")

    # Calculate INT8 model size from file
    import os
    if os.path.exists(quantized_model_path):
        int8_size = os.path.getsize(quantized_model_path) / 1024 / 1024
        print(f"INT8 Model size: {int8_size:.2f} MB")
        reduction = (1 - int8_size / fp32_size) * 100
        print(f"Size reduction: {reduction:.1f}%")
    else:
        print("INT8 model file not found")


def main():
    """Main function to run the complete QAT pipeline for CRNN"""
    # Configuration
    config = {
        'annotation_dir': 'D:/synlabs/SynlabsProject/ContainerOcr/train_images_full_size/labels/',
        'image_dir': 'D:/synlabs/SynlabsProject/ContainerOcr/train_images_full_size/images/',
        'weight_path': 'crnn_best.zip',
        'img_channel': 1,
        'img_height': 80,
        'img_width': 400,
        'batch_size': 4,
        'num_epochs': 5,
        'learning_rate': 1e-4,
        'map_to_seq_hidden': 64,
        'rnn_hidden': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'backend': 'fbgemm'  # Use 'fbgemm' for x86 CPUs
    }

    print(f"Using device: {config['device']}")
    print("Creating datasets...")

    # Create datasets
    train_dataset = OCRDataset(
        annotation_dir=config['annotation_dir'],
        image_dir=config['image_dir'],
        char_to_int=CHAR_TO_INT,
        img_size=(config['img_width'], config['img_height'])
    )

    print(f"Total training samples: {len(train_dataset)}")

    # Split dataset (80% train, 20% test)
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    # Create CRNN model with quantization support
    model = QuantizedCRNN(
        img_channel=config['img_channel'],
        img_height=config['img_height'],
        img_width=config['img_width'],
        num_class=NUM_CLASSES,
        map_to_seq_hidden=config['map_to_seq_hidden'],
        rnn_hidden=config['rnn_hidden']
    )

    print("Preparing CRNN model for QAT...")
    # Get example input for preparation
    example_batch = next(iter(train_loader))
    example_input = example_batch['images'][:1]  # Take first image

    # Prepare model for QAT
    model_qat = prepare_and_convert_model(model, example_input, config['backend'])

    print("Starting QAT training...")
    # Train with QAT
    model_trained = train_qat_model(
        model_qat,
        train_loader,
        weight_path=config['weight_path'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device']
    )

    print("Converting to INT8...")
    # Convert to INT8
    model_trained.eval()
    model_int8 = torch.ao.quantization.convert(model_trained)

    # Save the quantized model
    model_path = 'crnn_ocr_model_int8.pt'
    torch.jit.save(torch.jit.script(model_int8), model_path)
    print(f"Quantized CRNN model saved as '{model_path}'")

    print("Evaluating quantized model...")
    # Evaluate
    evaluate_model(model_int8, test_loader, device=config['device'])

    # Compare model sizes
    compare_model_sizes(model, model_path)

    print("CRNN QAT pipeline completed successfully!")


if __name__ == "__main__":
    main()
