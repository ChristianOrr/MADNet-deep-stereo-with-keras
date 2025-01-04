import wandb
import keras
import tensorflow as tf
from madnet import colorize_img

class WandBImagesCallback(keras.callbacks.Callback):
    """
    Logs image input data and disparity predictions
    to Weights and Biases dashboard. Can use the training data,
    validation data or both.

    Args:
        training_data: tf.data.Dataset object used for the .fit
            training.
        validation_data: tf.data.Dataset object used for .fit
            validation.
        val_epochs: Epoch frequency to log images to W and B dashboard.
    """
    def __init__(
            self,
            training_data=None,
            validation_data=None,
            val_epochs=1
    ):
        self.val_epochs = val_epochs
        self.training_data = None
        self.validation_data = None
        if training_data is not None:
            self.training_data = iter(training_data)
        if validation_data is not None:
            self.validation_data = iter(validation_data)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.val_epochs == 0:
            if self.training_data is not None:
                data = next(self.training_data)
                x, y = data
                shape = keras.ops.shape(y)
                if shape[0] > 1:
                    raise ValueError(f"Received batch_size {shape[0]} for training_data dataset. "
                                     "Please make sure batch size is 1")
                y_pred = self.model(x)
                if y is not None:
                    # Updates stateful loss metrics.
                    self.model.compute_loss(x, y, y_pred)
                    train_logs = self.model.compute_metrics(x, y, y_pred, None)
                    wandb.log({"Train": train_logs}, commit=True)
                train_images_dict = {
                    "Predicted Disparity": wandb.Image(colorize_img(y_pred, cmap='jet')[0].numpy()),
                    "Left Image": wandb.Image(x["left_input"].numpy()),
                    "Right Image": wandb.Image(x["right_input"].numpy())
                }
                if y is not None:
                    train_images_dict.update(
                        {"GroundTruth Disparity": wandb.Image(colorize_img(y, cmap="jet")[0].numpy())}
                    )
                wandb.log({"Train": train_images_dict}, commit=True)

            if self.validation_data is not None:
                data = next(self.validation_data)
                val_x, val_y = data
                shape = keras.ops.shape(val_y)
                if shape[0] > 1:
                    raise ValueError(f"Received batch_size {shape[0]} for validation_data dataset. "
                                     "Please make sure batch size is 1")
                val_y_pred = self.model(val_x)
                if val_y is not None:
                    # Updates stateful loss metrics.
                    self.model.compute_loss(val_x, val_y, val_y_pred)
                    val_logs = self.model.compute_metrics(val_x, val_y, val_y_pred, None)
                    wandb.log({"Val": val_logs}, commit=True)

                val_images_dict = {
                    "Predicted Disparity": wandb.Image(colorize_img(val_y_pred, cmap='jet')[0].numpy()),
                    "Left Image": wandb.Image(val_x["left_input"].numpy()),
                    "Right Image": wandb.Image(val_x["right_input"].numpy())
                }
                if val_y is not None:
                    val_images_dict.update(
                        {"GroundTruth Disparity": wandb.Image(colorize_img(val_y, cmap="jet")[0].numpy())}
                    )
                wandb.log({"Val": val_images_dict}, commit=True)


class TensorboardImagesCallback(keras.callbacks.Callback):
    """
    Logs image input data and disparity predictions
    to Tensorboard dashboard. Can use the training data,
    validation data or both.

    Args:
        training_data: tf.data.Dataset object used for the .fit
            training.
        validation_data: tf.data.Dataset object used for .fit
            validation.
        val_epochs: Epoch frequency to log images to Tensorboard.
    """
    def __init__(
            self,
            training_data=None,
            validation_data=None,
            val_epochs=1
    ):
        self.val_epochs = val_epochs
        self.training_data = None
        self.validation_data = None
        if training_data is not None:
            self.training_data = iter(training_data)
        if validation_data is not None:
            self.validation_data = iter(validation_data)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.val_epochs == 0:
            if self.training_data is not None:
                data = next(self.training_data)
                x, y, _ = data
                shape = keras.ops.shape(x["left_input"])
                if shape[0] > 1:
                    raise ValueError(f"Received batch_size {shape[0]} for training_data dataset. "
                                     "Please make sure batch size is 1")
                y_pred = self.model(x)
                if y is not None:
                    # Updates stateful loss metrics.
                    self.model.compute_loss(x, y, y_pred)
                    train_logs = self.model.compute_metrics(x, y, y_pred, None)
                    train_logs = {"train_" + name: val for name, val in train_logs.items()}
                    for key, value in train_logs.items():
                        tf.summary.scalar(name=key, data=value, step=epoch)

                tf.summary.image('train_01_predicted_disparity', colorize_img(y_pred, cmap='jet'),
                                 step=epoch, max_outputs=1)
                if y is not None:
                    tf.summary.image('train_02_groundtruth_disparity', colorize_img(y, cmap='jet'),
                                     step=epoch, max_outputs=1)
                tf.summary.image('train_03_left_image', x["left_input"], step=epoch, max_outputs=1)
                tf.summary.image('train_04_right_image', x["right_input"], step=epoch, max_outputs=1)

            if self.validation_data is not None:
                data = next(self.validation_data)
                val_x, val_y, _ = data
                shape = keras.ops.shape(val_x["left_input"])
                if shape[0] > 1:
                    raise ValueError(f"Received batch_size {shape[0]} for validation_data dataset. "
                                     "Please make sure batch size is 1")
                val_y_pred = self.model(val_x)
                if val_y is not None:
                    # Updates stateful loss metrics.
                    self.model.compute_loss(val_x, val_y, val_y_pred)
                    val_logs = self.model.compute_metrics(val_x, val_y, val_y_pred, None)
                    val_logs = {"val_" + name: val for name, val in val_logs.items()}
                    for key, value in val_logs.items():
                        tf.summary.scalar(name=key, data=value, step=epoch)

                tf.summary.image('val_01_predicted_disparity', colorize_img(val_y_pred, cmap='jet'),
                                 step=epoch, max_outputs=1)
                if val_y is not None:
                    tf.summary.image('val_02_groundtruth_disparity', colorize_img(val_y, cmap='jet'),
                                     step=epoch, max_outputs=1)
                tf.summary.image('val_03_left_image', val_x["left_input"], step=epoch, max_outputs=1)
                tf.summary.image('val_04_right_image', val_x["right_input"], step=epoch, max_outputs=1)


class TensorboardTestImagesCallback(keras.callbacks.Callback):
    """
    Logs image input data and disparity predictions
    to Tensorboard dashboard.

    Args:
        testing_data: tf.data.Dataset object used for the .fit
            testing.
        test_steps: Step frequency to log images to Tensorboard.
    """
    def __init__(
            self,
            testing_data=None,
            test_steps=1,
            pred_dir=None
    ):
        self.test_steps = test_steps
        self.testing_data = iter(testing_data)
        self.pred_dir = pred_dir

    def on_test_batch_end(self, batch, logs={}):
        if batch % self.test_steps == 0:
            data = next(self.testing_data)
            x, y = data
            shape = keras.ops.shape(x["left_input"])
            if shape[0] > 1:
                raise ValueError(f"Received batch_size {shape[0]} for testing_data dataset. "
                                 "Please make sure batch size is 1")
            y_pred = self.model(x)
            # Updates stateful loss metrics.
            self.model.compute_loss(x, y, y_pred)
            test_logs = self.model.compute_metrics(x, y, y_pred, None)
            test_logs = {"test_" + name: val for name, val in test_logs.items()}
            for key, value in test_logs.items():
                tf.summary.scalar(name=key, data=value, step=batch)
            pred_colorized = colorize_img(y_pred, cmap='jet')
            tf.summary.image('test_01_predicted_disparity', pred_colorized,
                             step=batch, max_outputs=20)
            tf.summary.image('test_02_groundtruth_disparity', colorize_img(y, cmap='jet'),
                             step=batch, max_outputs=20)
            tf.summary.image('test_03_left_image', x["left_input"], step=batch, max_outputs=20)
            tf.summary.image('test_04_right_image', x["right_input"], step=batch, max_outputs=20)
            if self.pred_dir is not None:
                image_path = self.pred_dir + f"/step{batch}.png"
                keras.utils.save_img(image_path, pred_colorized[0])
