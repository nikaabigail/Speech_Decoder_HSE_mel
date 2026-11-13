"""Historical CNN/BiLSTM regression encoder; state names are preserved."""
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    DOWNSAMPLING =  10
    HIDDEN_CHANNELS = 30
    FILTERING_SIZE = 25
    ENVELOPE_SIZE = 15
    FC_HIDDEN_FEATURES = 100

    assert FILTERING_SIZE % 2 == 1, "conv weights must be odd"
    assert ENVELOPE_SIZE % 2 == 1, "conv weights must be odd"

    def __init__(self, in_channels, out_channels, lag_backward, lag_forward, use_lstm=True):
        super(self.__class__, self).__init__()
        window_size = lag_backward + lag_forward + 1
        self.final_out_features = (window_size // self.DOWNSAMPLING + 1) * self.HIDDEN_CHANNELS
        self.use_lstm = use_lstm
        
        assert window_size > self.FILTERING_SIZE
        assert window_size > self.ENVELOPE_SIZE

        self.unmixing_layer = nn.Conv1d(in_channels, self.HIDDEN_CHANNELS, 1)
        self.unmixed_channels_batchnorm = torch.nn.BatchNorm1d(self.HIDDEN_CHANNELS, affine=False)
        self.detector = self._create_envelope_detector(self.HIDDEN_CHANNELS)
        self.features_batchnorm = torch.nn.BatchNorm1d(self.final_out_features, affine=False)
        self.lstm = nn.LSTM(self.HIDDEN_CHANNELS, int(self.HIDDEN_CHANNELS / 2), num_layers=1, batch_first=True, bidirectional=True)
        
        self.fc_layer = nn.Sequential(
            nn.Linear(self.final_out_features, out_channels),
        )

    def _create_envelope_detector(self, in_channels):  
        # 1. Learn band pass flter
        # 2. Centering signals
        # 3. Abs
        # 4. low pass to get envelope
        return nn.Sequential(
            nn.Conv1d(in_channels, in_channels,kernel_size=self.FILTERING_SIZE,
                      bias=False,  groups=in_channels,
                      padding=int(self.FILTERING_SIZE / 2)),
            nn.BatchNorm1d(in_channels, affine=False),
            nn.LeakyReLU(-1), # just absolute value
            nn.Conv1d(in_channels, in_channels, kernel_size=self.ENVELOPE_SIZE,
                      groups=in_channels, padding=int(self.ENVELOPE_SIZE / 2)),
        )

    def forward(self, x, return_hidden=False):
        self.unmixed_channels = self.unmixing_layer(x)
        unmixed_channels_scaled = self.unmixed_channels_batchnorm(self.unmixed_channels)
        detected_envelopes = self.detector(unmixed_channels_scaled)
        features = detected_envelopes[:, :, ::self.DOWNSAMPLING].contiguous()
        if self.use_lstm:
            features = self.lstm(features.transpose(1, 2))[0].transpose(1, 2)
        features = features.reshape((features.shape[0], -1))
        self.features_scaled = self.features_batchnorm(features)
        # hidden-state (Фаза 2.1): внутреннее представление перед последним FC.
        # Богаче декодированной мел; используется как вход word-классификатора.
        if return_hidden:
            return self.features_scaled
        output = self.fc_layer(self.features_scaled)
        return output
