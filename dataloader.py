import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image


class TextRecognitionDataset(Dataset):
    """ Class for working with training datasets in PyTorch. """

    def __init__(self, annotation_path, image_width, image_height, transform=None, shuffle=False):
        """
        Args:
            annotation_path (str): Path to annotation file
            image_width (int): Target image width
            image_height (int): Target image height
            transform (callable, optional): Optional transform to be applied
            shuffle (bool): Whether to shuffle the dataset
        """
        self.image_paths, self.labels = self.parse_datasets_arg(annotation_path)
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transform
        self.char_to_int, self.int_to_char, self.num_classes = self.create_character_maps()

        if shuffle:
            self.shuffle_data()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Read and preprocess image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # For vertical image
            image = image.transpose((1, 0))
            image = image[::-1]
            image = image.astype(np.float32)
            image = cv2.resize(image, (self.image_width, self.image_height))
            image = np.expand_dims(image, axis=0)  # Add channel dimension

            # Convert label to integer array
            label_int = np.array([self.char_to_int[char] for char in label.lower()], dtype=np.int32)

            if self.transform:
                image = self.transform(image)

            return torch.from_numpy(image), torch.from_numpy(label_int), label

        except Exception as e:
            print(f"Error processing image: {image_path}")
            raise e

    def shuffle_data(self):
        """Shuffles the dataset."""
        combined = list(zip(self.image_paths, self.labels))
        np.random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    @staticmethod
    def create_character_maps():
        """ Creates character-to-int and int-to-character maps. """
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        char_to_int = {char: i for i, char in enumerate(alphabet)}
        int_to_char = list(alphabet)
        return char_to_int, int_to_char, len(char_to_int) + 1

    @staticmethod
    def parse_datasets_arg(annotation_path):
        """ Parses datasets argument. """
        image_paths = []
        labels = []

        for ann_path in annotation_path.split(','):
            annotation_folder = os.path.dirname(ann_path)
            annotation_folder = os.path.join("/home/gaurav/path/to/clone/text_recognition", annotation_folder)

            with open(ann_path, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip().split()
                    if len(line) >= 2:  # Ensure we have both path and label
                        img_path = os.path.join(annotation_folder, line[0])
                        image_paths.append(img_path)
                        labels.append(line[1])

        return image_paths, labels

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length labels.
        Returns:
            images: Tensor of shape (batch_size, 1, H, W)
            labels: List of tensors with variable length
            label_strings: List of original label strings
        """
        images = torch.stack([item[0] for item in batch])
        labels = [item[1] for item in batch]
        label_strings = [item[2] for item in batch]

        return images, labels, label_strings


# Example usage:
if __name__ == "__main__":
    # Initialize dataset
    dataset = TextRecognitionDataset(
        annotation_path="path/to/annotations.txt",
        image_width=100,
        image_height=32
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=TextRecognitionDataset.collate_fn,
        num_workers=4
    )

    # Iterate through batches
    for images, labels, label_strings in dataloader:
        # images: (batch_size, 1, H, W)
        # labels: list of tensors with variable length
        # label_strings: list of original strings
        print(images.shape)
        print(len(labels))
        print(label_strings)
        break
