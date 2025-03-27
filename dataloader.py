import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order coordinates in consistent [top-left, top-right, bottom-right, bottom-left] order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image: np.ndarray, coords: List[int],
                         output_size: Tuple[int, int] = (100, 32)) -> np.ndarray:
    """
    Apply perspective transform to crop and straighten text region.

    Args:
        image: Input image (grayscale or color)
        coords: List of 4 coordinates [x1,y1,x2,y2]
        output_size: Desired output (width, height)

    Returns:
        Warped image
    """
    # Convert coordinates to 4 points
    x1, y1, x2, y2 = coords
    warped = image[y1:y2 , x1:x2 ]
    # pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    #
    # rect = order_points(np.array(pts))
    # (tl, tr, br, bl) = rect
    #
    # # Compute new width and height
    # widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # maxWidth = max(int(widthA), int(widthB))
    #
    # heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    # heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # maxHeight = max(int(heightA), int(heightB))
    #
    # # Destination points
    # dst = np.array([
    #     [0, 0],
    #     [maxWidth - 1, 0],
    #     [maxWidth - 1, maxHeight - 1],
    #     [0, maxHeight - 1]], dtype="float32")
    #
    # # Compute perspective transform and apply it
    # M = cv2.getPerspectiveTransform(rect, dst)
    # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    #
    # # Resize to desired output
    warped = cv2.resize(warped, output_size)
    return warped


class OCRDataset(Dataset):
    def __init__(self, annotation_dir: str, image_dir: str, char_to_int: Dict[str, int],
                 img_size: Tuple[int, int] = (100, 32), transform=None):
        """
        OCR Dataset for format: x1,y1,x2,y2,text

        Args:
            annotation_dir: Directory containing annotation files
            image_dir: Directory containing images
            char_to_int: Character mapping dictionary
            img_size: Output image size (width, height)
            transform: Optional transforms
        """
        self.annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.char_to_int = char_to_int
        self.img_width, self.img_height = img_size
        self.transform = transform

        # Pre-load all annotations with unique IDs
        self.annotations = []
        self.annotation_map = {}  # Maps annotation IDs to (file_idx, annotation_idx)

        for file_idx, file_name in enumerate(self.annotation_files):
            file_path = os.path.join(self.annotation_dir, file_name)
            file_annotations = self._load_annotation(file_path)

            for ann_idx, annotation in enumerate(file_annotations):
                unique_id = len(self.annotations)
                self.annotations.append(annotation)
                self.annotation_map[unique_id] = (file_idx, ann_idx)

    def _load_annotation(self, file_path: str) -> List[Dict]:
        """Load annotation from a single file in format: x1,y1,x2,y2,text"""
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    try:
                        x1, y1, x2, y2 = map(int, parts[:4])
                        text = ','.join(parts[4:]).strip()  # Handle text with commas
                        if text:  # Only add if text exists
                            annotations.append({
                                'coords': [x1, y1, x2, y2],
                                'text': text,
                                'points': [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # Convert to 4 points
                            })
                    except ValueError:
                        continue  # Skip malformed lines
        return annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        annotation = self.annotations[idx]
        file_idx, ann_idx = self.annotation_map[idx]
        file_name = self.annotation_files[file_idx]
        base_name = os.path.splitext(file_name)[0]

        # Load corresponding image
        img_path = os.path.join(self.image_dir, f"{base_name}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply perspective transform
        warped = four_point_transform(image, annotation['coords'],(self.img_width, self.img_height))
        cv2.imwrite("image.png" , warped )

        # Normalize and add channel dimension
        warped = warped.astype(np.float32) / 255.0
        warped = np.expand_dims(warped, axis=0)  # [1, H, W]

        # Convert text to indices
        text = annotation['text'].upper()  # Assuming case-insensitive
        text_int = torch.tensor(
            [self.char_to_int[c] for c in text if c in self.char_to_int],
            dtype=torch.long
        )

        if self.transform:
            warped = self.transform(warped)

        return {
            'image': torch.from_numpy(warped),
            'text': text,
            'text_int': text_int,  # This can be variable length
            'coords': torch.tensor(annotation['coords'], dtype=torch.float),
            'File_Name ' : base_name
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    images = torch.stack([item['image'] for item in batch])
    coords = torch.stack([item['coords'] for item in batch])

    # Handle variable-length text sequences
    texts = [item['text'] for item in batch]
    text_ints = [item['text_int'] for item in batch]

    # Get lengths of each sequence
    text_lens = torch.tensor([len(t) for t in text_ints], dtype=torch.long)

    # Pad sequences to maximum length in batch
    max_len = max(text_lens) if len(text_lens) > 0 else 0
    padded_text = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, text in enumerate(text_ints):
        if len(text) > 0:  # Only copy if text is not empty
            padded_text[i, :len(text)] = text

    return {
        'image': images,
        'text': texts,
        'text_int': padded_text,
        'text_len': text_lens,
        'coords': coords
    }


# Example usage
if __name__ == "__main__":
    # Create character mappings
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    char_to_int = {c: i for i, c in enumerate(chars)}

    # Initialize dataset
    dataset = OCRDataset(
        annotation_dir='D:/synlabs/contrain_ocr/Data/train_images/labels',
        image_dir='D:/synlabs/contrain_ocr/Data/train_images/images',
        char_to_int=char_to_int,
        img_size=(350//2, 180//2)
    )

    print(f"Total annotations: {len(dataset)}")

    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Example batch
    for batch in dataloader:
        print(batch['text_int'])
        print(f"\nBatch details:")
        print(f"Image batch shape: {batch['image'].shape}")  # Should be [batch_size, 1, H, W]
        print(f"Text samples: {batch['text']}")
        print(f"Padded text indices shape: {batch['text_int'].shape}")
        print(f"Text lengths: {batch['text_len']}")
        print(f"Coordinates shape: {batch['coords'].shape}")  # Should be [batch_size, 4]

        print("----------------------"*5)
        print("***"*10)
        break
