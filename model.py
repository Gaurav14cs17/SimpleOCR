import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRecognition(nn.Module):
    """ Text recognition model definition. """

    def __init__(self, is_training=True, num_classes=36, backbone_dropout=0.0):
        super(TextRecognition, self).__init__()
        self.is_training = is_training
        self.lstm_dim = 256
        self.num_classes = num_classes
        self.backbone_dropout = backbone_dropout

        # Feature extractor layers
        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout2d(self.backbone_dropout),
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(self.backbone_dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(self.backbone_dropout),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Dropout2d(self.backbone_dropout),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Dropout2d(self.backbone_dropout),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Dropout2d(self.backbone_dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Dropout2d(self.backbone_dropout),
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=512,
            hidden_size=self.lstm_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=2 * self.lstm_dim,  # *2 because bidirectional
            hidden_size=self.lstm_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Final classification layer
        self.fc = nn.Linear(2 * self.lstm_dim, self.num_classes)

    def forward(self, inputdata):
        features = self.feature_extractor(inputdata)
        logits = self.encoder_decoder(features)
        return logits

    def feature_extractor(self, inputdata):
        """ Extracts features from input text image. """
        # Add channel dimension if input is grayscale
        if inputdata.dim() == 3:
            inputdata = inputdata.unsqueeze(1)  # [B, 1, H, W]

        features = self.conv_layers(inputdata)

        # Squeeze height dimension (1 after conv layers)
        features = features.squeeze(2)  # [B, C, W]
        features = features.permute(0, 2, 1)  # [B, W, C]

        return features

    def encoder_decoder(self, inputdata):
        """ LSTM-based encoder-decoder module. """
        batch_size, width, _ = inputdata.size()

        # Encoder
        encoder_output, _ = self.encoder_lstm(inputdata)

        # Decoder
        decoder_output, _ = self.decoder_lstm(encoder_output)

        # Reshape for classification
        decoder_reshaped = decoder_output.contiguous().view(batch_size * width, -1)
        logits = self.fc(decoder_reshaped)
        logits = logits.view(batch_size, width, self.num_classes)

        # Transpose to [width, batch_size, num_classes] as in original
        rnn_out = logits.permute(1, 0, 2)
        return rnn_out

if __name__ == '__main__':
    # Parameters
    imgH = 32  # height of input image
    nc = 1  # number of input channels (1 for grayscale, 3 for color)
    nclass = 37  # number of classes (alphanumeric + blank)
    nh = 256  # size of LSTM hidden state

    # Create model
    model = TextRecognition()

    # Dummy input (batch_size, channels, height, width)
    input = torch.randn(1, nc, imgH, 100)

    # Forward pass
    output = model(input)
    print(f"Output shape: {output.shape}")  # (sequence_length, batch_size, nclass)
