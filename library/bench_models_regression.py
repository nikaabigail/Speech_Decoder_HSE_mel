"""Historical MEL40 and frozen Whisper-base L1-L6 configurations only."""
import torch
import numpy as np
from .models_regression import SimpleNet
from . import common_preprocessing, whisper_target
from .loggers import LearningLogStorer
from .runtime import DEVICE
from torch.utils.tensorboard import SummaryWriter


class BenchModelRegressionBase:
    BATCH_SIZE = 100
    DOWNSAMPED_APPROX_SAMPLING_RATE = 1000
    LEARNING_RATE = 0.0003
    
    HIGH_PASS_HZ = 10
    LOW_PASS_HZ = 200

    def __init__(self, patient):
        self.patient = patient
        self.calc_downsampling_coef()
        self.TEST_START_FILE_INDEX = self.patient["test_start_file_regression_index"]
        self.selected_channels = self.SELECTED_CHANNELS if self.SELECTED_CHANNELS is not None else self.patient["ecog_channels"]
        self.input_size = len(self.selected_channels)
        self.init()
        assert hasattr(self, "model")
        self.create_optimizer()
        self.create_logger()
        
    def init(self):
        # model
        raise NotImplementedError
        
    def calc_downsampling_coef(self,):
        self.downsampling_coef = round(self.patient["sampling_rate"] / self.DOWNSAMPED_APPROX_SAMPLING_RATE)
        
    def create_logger(self,):
        self.logger = LearningLogStorer(SummaryWriter(comment=f"___regression___{self.patient['name']}___{str(self.__class__.__name__)}"))
        
    def create_optimizer(self,):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        
    def preprocess_ecog(self, ecog):
        raise NotImplementedError

    def preprocess_sound(self, sound):
        raise NotImplementedError
        
    def detect_voice(self, y_batch):
        raise NotImplementedError


class SimpleNetBase(BenchModelRegressionBase):
    def init(self):
        self.model = SimpleNet(self.input_size, self.OUTPUT_SIZE, self.LAG_BACKWARD, self.LAG_FORWARD).to(DEVICE)

    def preprocess_ecog(self, ecog, sampling_rate):
        assert self.patient["sampling_rate"] == sampling_rate
        selected_channels = self.SELECTED_CHANNELS if self.SELECTED_CHANNELS is not None else self.patient["ecog_channels"]
        return common_preprocessing.classic_ecog_pipeline(ecog, self.patient["sampling_rate"], self.downsampling_coef, self.LOW_PASS_HZ, self.HIGH_PASS_HZ)[:, selected_channels]


class SimpleNetMelsBase(SimpleNetBase):

    def preprocess_sound(self, sound, sampling_rate, ecog_size):
        assert self.patient["sampling_rate"] == sampling_rate
        return common_preprocessing.classic_melspectrogram_pipeline(sound, self.patient["sampling_rate"], self.downsampling_coef, ecog_size, self.N_MELS, self.F_MAX)
    
    def detect_voice(self, y_batch):
        return np.sum(y_batch > 1, axis=1) > int(self.N_MELS * 0.25)


class SimpleNetMels40Base(SimpleNetMelsBase):
    N_MELS = 40
    F_MAX = 2000
    OUTPUT_SIZE = N_MELS


class SimpleNetCausalMels40Base(SimpleNetMels40Base):
    LAG_BACKWARD = 1000
    LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__40MELS(SimpleNetCausalMels40Base):
    SELECTED_CHANNELS = list(range(8, 16))


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__40MELS(SimpleNetCausalMels40Base):
    SELECTED_CHANNELS = list(range(6, 12))


class SimpleNetWhisperBase(SimpleNetBase):
    WHISPER_MODEL = "openai/whisper-base"
    WHISPER_LAYER = 4            # верхне-средний «speech»-tap (см. DESIGN.md); НЕ финальный
    WHISPER_D = 512             # Whisper-base embedding dimension
    USE_PCA = True
    PCA_COMPONENTS = 50
    OUTPUT_SIZE = 50            # PCA target dimension
    IS_WHISPER_TARGET = True

    def preprocess_sound(self, sound, sampling_rate, ecog_size):
        assert self.patient["sampling_rate"] == sampling_rate
        return whisper_target.classic_whisper_pipeline(
            sound, self.patient["sampling_rate"], self.downsampling_coef, ecog_size,
            self.WHISPER_LAYER, self.WHISPER_MODEL)

    def detect_voice(self, y_batch):
        # речевой канал нейтрализован -> correlation_speech == correlation; отбор по correlation
        return np.ones(len(y_batch), dtype=bool)


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__WHISPER_BASE_L4(SimpleNetWhisperBase):
    SELECTED_CHANNELS = list(range(6, 12)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__WHISPER_BASE_L4(SimpleNetWhisperBase):
    SELECTED_CHANNELS = list(range(8, 16)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__WHISPER_BASE_L1(SimpleNetWhisperBase):
    WHISPER_LAYER = 1
    SELECTED_CHANNELS = list(range(8, 16)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__WHISPER_BASE_L2(SimpleNetWhisperBase):
    WHISPER_LAYER = 2
    SELECTED_CHANNELS = list(range(8, 16)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__WHISPER_BASE_L3(SimpleNetWhisperBase):
    WHISPER_LAYER = 3
    SELECTED_CHANNELS = list(range(8, 16)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__WHISPER_BASE_L5(SimpleNetWhisperBase):
    WHISPER_LAYER = 5
    SELECTED_CHANNELS = list(range(8, 16)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__WHISPER_BASE_L6(SimpleNetWhisperBase):
    WHISPER_LAYER = 6
    SELECTED_CHANNELS = list(range(8, 16)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__WHISPER_BASE_L1(SimpleNetWhisperBase):
    WHISPER_LAYER = 1
    SELECTED_CHANNELS = list(range(6, 12)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__WHISPER_BASE_L2(SimpleNetWhisperBase):
    WHISPER_LAYER = 2
    SELECTED_CHANNELS = list(range(6, 12)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__WHISPER_BASE_L3(SimpleNetWhisperBase):
    WHISPER_LAYER = 3
    SELECTED_CHANNELS = list(range(6, 12)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__WHISPER_BASE_L5(SimpleNetWhisperBase):
    WHISPER_LAYER = 5
    SELECTED_CHANNELS = list(range(6, 12)); LAG_BACKWARD = 1000; LAG_FORWARD = 0


class SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__WHISPER_BASE_L6(SimpleNetWhisperBase):
    WHISPER_LAYER = 6
    SELECTED_CHANNELS = list(range(6, 12)); LAG_BACKWARD = 1000; LAG_FORWARD = 0
