import os
import tensorflow as tf
import argparse
from madnet import MADNet
from preprocessing import StereoDatasetCreator
from losses_and_metrics import Bad3, EndPointError, ReconstructionLoss, SSIMLoss
from callbacks import TensorboardImagesCallback


parser = argparse.ArgumentParser(description='Script for training MADNet')
parser.add_argument("--train_left_dir", help='path to left images folder', required=True)
parser.add_argument("--train_right_dir", help='path to right images folder', required=True)
parser.add_argument("--train_disp_dir", help='path to left disparity maps folder', default=None, required=False)
parser.add_argument("--val_left_dir", help='path to left images folder', default=None, required=False)
parser.add_argument("--val_right_dir", help='path to right images folder', default=None, required=False)
parser.add_argument("--val_disp_dir", help='path to left disparity maps folder', default=None, required=False)
parser.add_argument("--shuffle", help='shuffle training dataset', action="store_true", default=False)
parser.add_argument("--search_range", help='maximum dispacement (ie. smallest disparity)',
                    default=2, type=int, required=False)
parser.add_argument("-o", "--output_dir",
                    help='path to folder for outputting tensorboard logs and saving model weights',
                    required=True)
parser.add_argument("--checkpoint_path",
                    help="path to pretrained MADNet checkpoint file (for fine turning)",
                    default=None, required=False)
parser.add_argument("--lr", help="initial value for learning rate",default=0.0001, type=float, required=False)
parser.add_argument("--height", help='model image input height resolution', type=int, default=480)
parser.add_argument("--width", help='model image input height resolution', type=int, default=640)
parser.add_argument("--batch_size", help='batch size to use during training',type=int,default=1)
parser.add_argument("--num_epochs", help='number of training epochs', type=int, default=1000)
parser.add_argument("--epoch_steps", help='training steps per epoch', type=int, default=1000)
parser.add_argument("--save_freq", help='model saving frequncy per steps', type=int, default=1000)
parser.add_argument("--epoch_evals", help='number of epochs per evaluation', type=int, default=1)
parser.add_argument("--eval_steps", help='number of batches to process per evaluation', type=int, default=1)
parser.add_argument("--log_tensorboard", help="Logs results to tensorboard events files.", action="store_true")
args = parser.parse_args()


def main(args):
    perform_val = False
    if args.val_left_dir is not None and args.val_right_dir is not None and args.val_disp_dir is not None:
        perform_val = True
    # Create output folder if it doesn't already exist
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = args.output_dir + "/logs"

    # Initialise the model
    model = MADNet(
        input_shape=(args.height, args.width, 3),
        weights=args.checkpoint_path,
        search_range=args.search_range
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    # If no train groundtruth is available, then the reprojection error
    # from warping is used to calculate the loss
    if args.train_disp_dir is None:
        model.compile(
            optimizer=optimizer,
            loss=SSIMLoss(),
            metrics=[EndPointError(), Bad3()],
            run_eagerly=True if perform_val else False
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=ReconstructionLoss(),
            metrics=[EndPointError(), Bad3()],
            run_eagerly=False
        )

    # Get training data
    train_dataset = StereoDatasetCreator(
        left_dir=args.train_left_dir,
        right_dir=args.train_right_dir,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        shuffle=args.shuffle,
        disp_dir=args.train_disp_dir
    )
    train_ds = train_dataset().repeat()
    # Get validation data
    val_ds = None
    if perform_val:
        val_dataset = StereoDatasetCreator(
            left_dir=args.val_left_dir,
            right_dir=args.val_right_dir,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            shuffle=args.shuffle,
            disp_dir=args.val_disp_dir
        )
        val_ds = val_dataset().repeat()

    # Create callbacks
    def scheduler(epoch, lr):
        if epoch < 30000:
            lr = lr
        elif epoch < 60000:
            lr = lr * 0.5
        else:
            # learning_rate * decay_rate ^ (global_step / decay_steps)
            decay_rate = 0.5
            lr = lr * 0.5 * decay_rate ** (epoch // 60000)
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        return lr
    schedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.output_dir + "/checkpoints/epoch-{epoch:04d}.ckpt",
        save_freq=args.save_freq,
        save_weights_only=True,
        verbose=0
    )
    all_callbacks = [
            save_callback,
            schedule_callback
        ]
    if args.log_tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_steps_per_second=True,
            update_freq="batch"
        )
        all_callbacks.append(tensorboard_callback)
        tensorboard_images_callback = TensorboardImagesCallback(
            training_data=train_ds,
            validation_data=val_ds,
            val_epochs=args.epoch_evals
        )
        all_callbacks.append(tensorboard_images_callback)
    # Fit the model
    history = model.fit(
        x=train_ds,
        epochs=args.num_epochs,
        verbose=1,
        steps_per_epoch=args.epoch_steps,
        callbacks=all_callbacks
    )


if __name__ == "__main__":
    main(args)