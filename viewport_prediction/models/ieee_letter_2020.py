from tensorflow import keras as tfk
from tensorflow.keras.layers import (  # pylint: disable=no-name-in-module,import-error
    LSTM,
    Dense,
    TimeDistributed,
    GlobalAveragePooling2D,
)

from viewport_prediction.data.hf_dataset import HFDataset
from viewport_prediction.models.base_model import BaseModel
from viewport_prediction.config.experiment_config import BaseModelConfig


class IEEELetter2020ModelConfig(BaseModelConfig):
    video_frame_shape: list[int]


class IEEELetter2020Model(BaseModel[IEEELetter2020ModelConfig]):
    """Model [1].

    References:
    [1] X. Chen, A. T. Z. Kasgari and W. Saad, "Deep Learning for Content-Based
    Personalized Viewport Prediction of 360-Degree VR Videos," in IEEE Networking Letters,
    vol. 2, no. 2, pp. 81-84, June 2020, doi: 10.1109/LNET.2020.2977124.
    """

    Config = IEEELetter2020ModelConfig
    Dataset = HFDataset

    def build(self) -> None:
        model_config = self.config.model

        # inputs
        input_video_frame = tfk.Input(
            shape=(model_config.past_window_size, *model_config.video_frame_shape),
            name="input_video_frame",
        )
        input_head_orientation = tfk.Input(
            shape=(model_config.past_window_size, 2),
            name="input_head_orientation",
        )

        # sub-model to extract video frame features
        mobile_net = tfk.applications.MobileNetV2(
            input_shape=model_config.video_frame_shape,
            weights="imagenet",
            include_top=False,
        )
        mobile_net_output = mobile_net.output
        mobile_net_output = GlobalAveragePooling2D()(mobile_net_output)
        mobile_net_output = Dense(1024, activation="relu")(mobile_net_output)
        mobile_net_output = Dense(512, activation="relu")(mobile_net_output)
        mobile_net_output = Dense(32, activation="relu")(mobile_net_output)
        video_frame_feat_model = tfk.Model(
            inputs=mobile_net.input,
            outputs=mobile_net_output,
        )

        # concat features
        head_orientation_feat = TimeDistributed(Dense(32))(input_head_orientation)
        video_frame_feat = TimeDistributed(video_frame_feat_model)(input_video_frame)
        feat = tfk.layers.Concatenate(axis=2)([head_orientation_feat, video_frame_feat])

        # prediction module
        output = LSTM(units=256, return_state=True, return_sequences=True)(feat)
        output = LSTM(units=256, return_sequences=True)(feat)
        output = TimeDistributed(Dense(units=2))(feat)

        # create model
        self.model = tfk.Model(
            inputs=[
                input_head_orientation,
                input_video_frame,
            ],
            outputs=output,
        )

        # freeze mobilenet layers
        for layer in mobile_net.layers:
            layer.trainable = False
