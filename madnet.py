import random
import tensorflow as tf
import keras
from keras import backend
from keras.utils import get_file
from keras.utils import get_source_inputs
from keras import layers
import numpy as np
from matplotlib import cm


def colorize_img(value, vmin=None, vmax=None, cmap='jet'):
    """
    A utility function for TensorFlow that maps a grayscale image to
    a matplotlib colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1
    before mapping to a grayscale colormap.
    Arguments:
      - value: 4D Tensor of shape [batch_size, height, width, 1]
      - vmin: the minimum value of the range used for normalization. (Default: value minimum)
      - vmax: the maximum value of the range used for normalization. (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's 'get_cmap'.(Default: 'gray')
    
    Returns a 3D tensor of shape [batch_size,height, width,3].
    """
    # normalize
    vmin = keras.ops.min(value) if vmin is None else vmin
    vmax = keras.ops.max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # quantize
    indices = keras.ops.cast(keras.ops.round(value[:, :, :, 0]*255), dtype="int32")

    # gather
    color_map = cm.get_cmap(cmap)
    colors = color_map(np.arange(256))[:, :3]
    colors = keras.ops.convert_to_tensor(colors, dtype="float32")
    value = keras.ops.take(colors, indices)
    return value


# https://github.com/philferriere/tfoptflow/blob/bdc7a72e78008d1cd6db46e4667dffc2bab1fe9e/tfoptflow/core_costvol.py
class CostVolumeBlock(keras.layers.Layer):
    """Build cost volume for associating a pixel from the
    left image with its corresponding pixels in the right image.
    Args:
        c1: Level of the feature pyramid of the left image
        warp: Warped level of the feature pyramid of the right image
        search_range: Search range (maximum displacement)
    """
    def __init__(self, search_range):
        super(CostVolumeBlock, self).__init__()
        self.search_range = search_range

    def call(self, c1, warp):
        padded_lvl = keras.ops.pad(warp, [[0, 0], [0, 0], [self.search_range, self.search_range], [0, 0]])
        width = c1.shape[2]
        max_offset = self.search_range * 2 + 1

        cost_vol = []
        for i in range(0, max_offset):
            slice = padded_lvl[:, :, i:width+i, :]
            cost = keras.ops.mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)

        cost_vol = keras.ops.concatenate(cost_vol, axis=3)
        cost_curve = keras.ops.concatenate([c1, cost_vol], axis=3)

        return cost_curve


class BilinearSampler(keras.layers.Layer):
    """
    Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.
    Args:
        imgs: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,width_t, 2].
                height_t/width_t correspond to the dimensions of the output image
                (don't need to be the same as height_s/width_s). The two channels
                correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels]
    """
    def __init__(self):
        super(BilinearSampler, self).__init__()

    def _repeat(self, x, n_repeats):
        rep = keras.ops.transpose(
            keras.ops.expand_dims(keras.ops.ones(shape=keras.ops.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = keras.ops.cast(rep, 'float32')
        x = keras.ops.matmul(keras.ops.reshape(x, (-1, 1)), rep)
        return keras.ops.reshape(x, [-1])

    def call(self, imgs, coords):
        coords_x, coords_y = keras.ops.split(coords, 2, axis=3)
        inp_size = keras.ops.shape(imgs)
        coord_size = keras.ops.shape(coords)
        out_size = [coord_size[0], coord_size[1], coord_size[2], inp_size[3]]

        coords_x = keras.ops.cast(coords_x, 'float32')
        coords_y = keras.ops.cast(coords_y, 'float32')

        x0 = keras.ops.floor(coords_x)
        x1 = x0 + 1
        y0 = keras.ops.floor(coords_y)
        y1 = y0 + 1

        y_max = keras.ops.cast(inp_size[1] - 1, 'float32')
        x_max = keras.ops.cast(inp_size[2] - 1, 'float32')
        zero = keras.ops.zeros([1], dtype='float32')

        wt_x0 = x1 - coords_x
        wt_x1 = coords_x - x0
        wt_y0 = y1 - coords_y
        wt_y1 = coords_y - y0

        x0_safe = keras.ops.clip(x0, zero[0], x_max)
        y0_safe = keras.ops.clip(y0, zero[0], y_max)
        x1_safe = keras.ops.clip(x1, zero[0], x_max)
        y1_safe = keras.ops.clip(y1, zero[0], y_max)

        ## indices in the flat image to sample from
        dim2 = keras.ops.cast(inp_size[2], 'float32')
        dim1 = keras.ops.cast(inp_size[2] * inp_size[1], 'float32')
        base = keras.ops.reshape(
            self._repeat(keras.ops.cast(keras.ops.arange(coord_size[0]), 'float32') * dim1, coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1]
        )

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = x0_safe + base_y0
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        im00 = keras.ops.take(imgs, keras.ops.cast(idx00, 'int32'))
        im01 = keras.ops.take(imgs, keras.ops.cast(idx01, 'int32'))
        im10 = keras.ops.take(imgs, keras.ops.cast(idx10, 'int32'))
        im11 = keras.ops.take(imgs, keras.ops.cast(idx11, 'int32'))

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = keras.ops.add(
            keras.ops.add(w00 * im00, w01 * im01),
            keras.ops.add(w10 * im10, w11 * im11)
        )

        return output

class WarpImageBlock(keras.layers.Layer):
    """
    Given an image and a flow generate the warped image,
    for stereo img is the right image, flow is the disparity aligned with left.
    img: image that needs to be warped
    flow: Generic optical flow or disparity
    """

    def __init__(self):
        super(WarpImageBlock, self).__init__()
        self.bilinear_sampler = BilinearSampler()

    def build_coords(self, immy):
        max_height = 2048
        max_width = 2048
        pixel_coords = np.ones((1, max_height, max_width, 2))

        # build pixel coordinates and their disparity
        for i in range(0, max_height):
            for j in range(0, max_width):
                pixel_coords[0][i][j][0] = j
                pixel_coords[0][i][j][1] = i

        pixel_coords = keras.ops.convert_to_tensor(pixel_coords, "float32")
        real_height = keras.ops.shape(immy)[1]
        real_width = keras.ops.shape(immy)[2]
        real_pixel_coord = pixel_coords[:, 0:real_height, 0:real_width, :]
        immy = keras.ops.concatenate([immy, keras.ops.zeros_like(immy)], axis=-1)
        output = real_pixel_coord - immy

        return output

    def call(self, img, flow):

        coords = self.build_coords(flow)
        warped = self.bilinear_sampler(img, coords)
        return warped


class RefinementBlock(keras.layers.Layer):
    """
    Final Layer in MADNet.
    Calculates the reprojection loss if training=True.
    Args:
        input: left_F2 tensor
        disp: D2 disparity from M2 module
        final_left: full resolution RGB left image
        final_right: full resolution RGB right image
    Returns:
        Full resolution disparity in float32 normalized 0-1
    """
    def __init__(self, output_shape):
        super(RefinementBlock, self).__init__()
        self.output_shape = output_shape
        layer_kwargs = {
            "kernel_size": (3, 3),
            "padding": "same",
            "activation": keras.layers.Activation(keras.activations.leaky_relu, dtype="float32", name="leaky_relu"),
            "use_bias": True
        }
        self.context1 = keras.layers.Conv2D(filters=128, dilation_rate=1, name="context1", **layer_kwargs)
        self.context2 = keras.layers.Conv2D(filters=128, dilation_rate=2, name="context2", **layer_kwargs)
        self.context3 = keras.layers.Conv2D(filters=128, dilation_rate=4, name="context3", **layer_kwargs)
        self.context4 = keras.layers.Conv2D(filters=96, dilation_rate=8, name="context4", **layer_kwargs)
        self.context5 = keras.layers.Conv2D(filters=64, dilation_rate=16, name="context5", **layer_kwargs)
        self.context6 = keras.layers.Conv2D(filters=32, dilation_rate=1, name="context6", **layer_kwargs)
        self.context7 = keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            dilation_rate=1,
            padding="same",
            activation="linear",
            use_bias=True,
            name="context7"
        )

    def call(self, input, disp):
        volume = keras.ops.concatenate([input, disp], axis=-1)
        x = self.context1(volume)
        x = self.context2(x)
        x = self.context3(x)
        x = self.context4(x)
        x = self.context5(x)
        x = self.context6(x)
        x = self.context7(x)

        context_disp = keras.ops.add(disp, x)
        final_disparity = keras.ops.image.resize(
            images=context_disp,
            size=(self.output_shape[0], self.output_shape[1]),
            interpolation='bilinear'
        )
        return final_disparity


class StereoEstimatorBlock(keras.layers.Layer):
    """
    This is the stereo estimation network at resolution n.
    It uses the costs (from the pixel difference between the warped right image 
    and the left image) combined with the upsampled disparity from the previous
    layer (when the layer is not the last layer).

    The output is predicted disparity for the network at resolution n.
    """
    def __init__(self, name):
        super(StereoEstimatorBlock, self).__init__()
        layer_kwargs = {
            "kernel_size": (3, 3),
            "strides": 1,
            "padding": "same",
            "activation": keras.layers.Activation(keras.activations.leaky_relu, dtype="float32", name="leaky_relu"),
            "use_bias": True
        }
        self.disp1 = keras.layers.Conv2D(filters=128, name=f"{name}_disp1", **layer_kwargs)
        self.disp2 = keras.layers.Conv2D(filters=128, name=f"{name}_disp2", **layer_kwargs)
        self.disp3 = keras.layers.Conv2D(filters=96, name=f"{name}_disp3", **layer_kwargs)
        self.disp4 = keras.layers.Conv2D(filters=64, name=f"{name}_disp4", **layer_kwargs)
        self.disp5 = keras.layers.Conv2D(filters=32, name=f"{name}_disp5", **layer_kwargs)
        self.disp6 = keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="linear",
            use_bias=True,
            name=f"{name}_disp6"
        )

    def call(self, costs, upsampled_disp=None):
        if upsampled_disp is not None:
            volume = keras.ops.concatenate([costs, upsampled_disp], axis=-1)
        else:
            volume = costs

        x = self.disp1(volume)
        x = self.disp2(x)
        x = self.disp3(x)
        x = self.disp4(x)
        x = self.disp5(x)
        x = self.disp6(x)
        return x


class ModuleM(keras.layers.Layer):
    """
    Module MX is a sub-module of MADNet, which can be trained individually for 
    online adaptation using the MAD (Modular ADaptaion) method.
    """
    def __init__(self, layer, search_range):
        super(ModuleM, self).__init__()
        self.warp_image_block = WarpImageBlock()
        self.layer = layer
        self.search_range = search_range
        self.cost_volume_block = CostVolumeBlock(search_range=search_range)
        self.stereo_estimator_block = StereoEstimatorBlock(name=f"volume_filtering_{self.layer}")

    def call(self, inputs):
        # Check if layer is the bottom of the pyramid
        if len(inputs) == 3:
            left, right, prev_disp = inputs
            mod_height, mod_width = left.shape[1], left.shape[2]
            # Upsample disparity from previous layer
            upsampled_disp = keras.ops.image.resize(
                images=prev_disp,
                size=(mod_height, mod_width),
                interpolation='bilinear'
            )
            # Warp the right image into the left using upsampled disparity
            warped_left = self.warp_image_block(right, upsampled_disp)
        else:
            left, right = inputs
            # No previous disparity exits, so use right image instead of warped left
            warped_left = right

        costs = self.cost_volume_block(left, warped_left)

        # Get the disparity using cost volume between left and warped left images
        if len(inputs) == 3:
            module_disparity = self.stereo_estimator_block(costs, upsampled_disp)
        else:
            module_disparity = self.stereo_estimator_block(costs)

        return module_disparity


@tf.function
def _custom_train_step(self, data):
    """
    This is a monkey patch for the standard keras train_step.

    This patch adds the following training features:
        1. Training without groundtruth disparity. (self-supervised training)
        2. Tensorboard summaries.
        3. Loss is reduced for batch sizes larger than 1.
    """
    # Left, right image inputs and groundtruth target disparity
    inputs, gt, sample_weight = data
    left_input = inputs["left_input"]
    right_input = inputs["right_input"]

    with tf.GradientTape(persistent=False) as tape:
        # Forward pass
        final_disparity = self(inputs=inputs, training=True)
        # Calculate loss
        if gt is None:
            # Warp the right image into the left using final disparity
            warped_left = WarpImageBlock()(right_input, final_disparity)
            loss = self.compute_loss(left_input, warped_left, sample_weight)
        else:
            loss = self.compute_loss(gt, final_disparity, sample_weight)
        # Perform reduction on the loss
        # Note: displayed loss will be sum of all batch losses, but backprop will use the reduced loss
        batch_size = keras.ops.shape(left_input)[0]
        reduced_loss = loss / keras.ops.cast(batch_size, dtype="float32")

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(reduced_loss, trainable_vars)   

    # Run backwards pass.
    self.optimizer.apply(gradients, trainable_vars)

    # return_metrics = {}
    if gt is not None:
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(gt, final_disparity, sample_weight)

    return {m.name: m.result() for m in self.metrics}


def _custom_test_step(predict_func):

    @tf.function
    def _test_step_block(self, data):
        inputs, gt, sample_weight = data
        y_pred = predict_func(self, data)
        # Updates stateful loss metrics.
        return_metrics = {}
        self.compiled_metrics.reset_state()
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics
    return _test_step_block


def _custom_predict_step(num_adapt, mad_type):
    """
    This is a monkey patch for the standard keras predict_step.

    A Closure is utilised to enable the different inferencing modes shown below.
    This patch adds the following inferencing options:
        1. Full adaptation while inferencing. (self-supervised learning)
        2. or MAD adapation while inferencing. With options to adapt
           between 1-5 modules. (also self-supervised learning, but slower learning)
    """
    # Full backprop on all layers
    if num_adapt == 6:
        @tf.function
        def _predict_step_block(self, data):
            inputs, _, _ = data

            left_input = inputs["left_input"]
            right_input = inputs["right_input"]

            with tf.GradientTape(persistent=False) as tape:
                # Forward pass
                final_disparity = self(inputs=inputs, training=True)
                # Calculate loss
                # Warp the right image into the left using final disparity
                warped_left = WarpImageBlock()(right_input, final_disparity)
                loss = self.compute_loss(left_input, warped_left)

                # Perform reduction on the loss
                # Note: displayed loss will be sum of all batch losses, but backprop will use the reduced loss
                batch_size = keras.ops.shape(left_input)[0]
                reduced_loss = loss / keras.ops.cast(batch_size, dtype="float32")

            # Run backwards pass.
            self.optimizer.minimize(reduced_loss, self.trainable_variables, tape=tape)
            return final_disparity
    # MAD adaptation
    else:
        @tf.function
        def _predict_step_block(self, data):
            module_layers = [
                ["conv1", "conv2",
                 "context1", "context2", "context3", "context4", "context5", "context6", "context7"],
                ["conv3", "conv4",
                 "volume_filtering_2_disp1", "volume_filtering_2_disp2", "volume_filtering_2_disp3",
                 "volume_filtering_2_disp4", "volume_filtering_2_disp5", "volume_filtering_2_disp6"],
                ["conv5", "conv6",
                 "volume_filtering_3_disp1", "volume_filtering_3_disp2", "volume_filtering_3_disp3",
                 "volume_filtering_3_disp4", "volume_filtering_3_disp5", "volume_filtering_3_disp6"],
                ["conv7", "conv8",
                 "volume_filtering_4_disp1", "volume_filtering_4_disp2", "volume_filtering_4_disp3",
                 "volume_filtering_4_disp4", "volume_filtering_4_disp5", "volume_filtering_4_disp6"],
                ["conv9", "conv10",
                 "volume_filtering_5_disp1", "volume_filtering_5_disp2", "volume_filtering_5_disp3",
                 "volume_filtering_5_disp4", "volume_filtering_5_disp5", "volume_filtering_5_disp6"],
                ["conv11", "conv12",
                 "volume_filtering_6_disp1", "volume_filtering_6_disp2", "volume_filtering_6_disp3",
                 "volume_filtering_6_disp4", "volume_filtering_6_disp5", "volume_filtering_6_disp6"],
            ]
            inputs, _, _ = data

            left_input = inputs["left_input"]
            right_input = inputs["right_input"]

            with tf.GradientTape(persistent=True) as tape:
                # Forward pass
                final_disparity = self(inputs=inputs, training=True)
                # Calculate loss
                # Warp the right image into the left using final disparity
                warped_left = WarpImageBlock()(right_input, final_disparity)
                loss = self.compute_loss(left_input, warped_left)

                # Perform reduction on the loss
                # Note: displayed loss will be sum of all batch losses, but backprop will use the reduced loss
                batch_size = keras.ops.shape(left_input)[0]
                reduced_loss = loss / keras.ops.cast(batch_size, dtype="float32")

            # Run backwards pass.
            if mad_type == "random":
                # adapt_modules = random.sample(list(module_layers_dict.keys()), num_adapt)
                adapt_modules = random.sample(range(6), num_adapt)
            elif mad_type == "sequential":
                adapt_modules = []
                for i in range(num_adapt):
                    new_id = i + self.last_adapt
                    if new_id > 5:
                        new_id = new_id % 6
                    adapt_modules.append(new_id)
                self.last_adapt.assign(new_id)

            all_vars = [[], [], [], [], [], []]
            for i in range(6):
                for layer in module_layers[i]:
                    model_layer = self.get_layer(layer)
                    layer_vars = model_layer.trainable_variables
                    for var in layer_vars:
                        all_vars[i].append(var)

            # Graph tracing requires all variables to be created on the first pass,
            # so performing full mad on first pass
            if self.first_adapt_pass:
                adapt_modules = range(6)
                self.first_adapt_pass = False

            if mad_type == "random":
                # this adaptation method is faster but doesn't work with sequential
                for i in range(6):
                    if i in adapt_modules:
                        self.optimizer.minimize(reduced_loss, all_vars[i], tape=tape)
            elif mad_type == "sequential":
                def do_nothing(loss, vars, tape):
                    # function that mimics the inputs and outputs of the optimize function
                    return keras.ops.array([True, True, True, True, True, True])

                for i in range(6):
                    keras.ops.cond(
                        keras.ops.any(keras.ops.equal(i, adapt_modules)),
                        true_fn=lambda: self.optimizer.minimize(reduced_loss, all_vars[i], tape=tape),
                        false_fn=lambda: do_nothing(reduced_loss, all_vars[i], tape=tape)
                    )

            return final_disparity

    return _predict_step_block


def MADNet(input_shape=None,
           weights=None,
           input_tensor=None,
           num_adapt_modules=0,
           mad_mode="random",
           search_range=2
           ):
    pretrained_weights = {"synthetic", "kitti", "tf1_conversion_synthetic", "tf1_conversion_kitti"}
    f"""
    Instantiates the MADNet architecture

    Reference:
        - [MADNet: Real-time self-adaptive deep stereo](
          https://arxiv.org/abs/1810.05424) (CVPR 2019)

    Args:
        input_shape: Optional shape tuple, to be specified if you would
            like to use a model with an input image resolution that is not
            (480, 640, 3).
            It should have exactly 3 inputs channels (480, 640, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
        weights: String, one of `None` (random initialization),
            or one of the following pretrained weights: {pretrained_weights},
            or the path to the weights file to be loaded.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        num_adapt_modules: Integer, number of modules to perform adaptation on while inferencing.
            For standard inferencing, use num_adapt_modules=0,
            MAD is num_adapt_modules=2-5,
            Full backprop is num_adapt_modules=6.
            Note: This is for inferencing only, so doesnt affect training.
            If you would like to change the inferencing mode you will need to
            instantiate the model again with the new num_adapt_modules value.
        mad_mode: String, one of "random" or "sequential"
            This is only needed for MAD adaptation with num_adapt_modules in 1-5.
            "random", selects the modules to adapt randomly.
            "sequential", selects the modules to adapt sequentially. 
        search_range: maximum search displacement for the cost volume

    Returns:
        A `keras.Model` instance.
    """
    if not (weights is None or
            weights in pretrained_weights or
            tf.io.gfile.exists(weights) or
            tf.io.gfile.exists(weights + ".index")):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         f'one of the following pretrained weights: {pretrained_weights}, '
                         'or the path to the weights file to be loaded. \n'
                         f'Received `weights={weights}`')
    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError(
                    f'input_tensor: {input_tensor}'
                    'is not type input_tensor. '
                    f'Received `type(input_tensor)={type(input_tensor)}`'
                )
        if is_input_t_tensor:
            if backend.image_data_format() == 'channels_first':
                raise ValueError('Detected input_tensor in channels_first mode '
                                 'please ensure channels are last`; '
                                 'Received `input_tensor.shape='
                                 f'{input_tensor.shape}')
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError(
                        'input_tensor.shape[2] must equal input_shape[1]; '
                        'Received `input_tensor.shape='
                        f'{input_tensor.shape}`, '
                        f'`input_shape={input_shape}`')
        else:
            raise ValueError('input_tensor is not a Keras tensor; '
                             f'Received `input_tensor={input_tensor}`')


    default_shape = (480, 640, 3)
    # If input_shape is None, infer shape from input_tensor.
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor must be a valid Keras tensor type; '
                             f'Received {input_tensor} of type {type(input_tensor)}')

        if input_shape is None and not backend.is_keras_tensor(input_tensor):
            input_shape = default_shape
        elif input_shape is None and backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == 'channels_first':
                raise ValueError('Detected input_tensor in channels_first mode '
                                 'please ensure channels are last`; '
                                 'Received `input_tensor.shape='
                                 f'{input_tensor.shape}')
            else:
                input_shape = (backend.int_shape(input_tensor)[1],
                               backend.int_shape(input_tensor)[2],
                               3)

    # If input_shape is None and no input_tensor
    elif input_shape is None:
        input_shape = default_shape

    # If input_shape is not None, assume default size.
    else:
        if backend.image_data_format() == 'channels_first':
            raise ValueError('Detected input_tensor in channels_first mode '
                             'please ensure channels are last`; '
                             'Received `input_tensor.shape='
                             f'{input_tensor.shape}')

    if type(num_adapt_modules) is not int or num_adapt_modules < 0 or num_adapt_modules > 6:
        raise ValueError("num_adapt_modules needs to be an integer from 0-6."
                         f"\nDetected num_adapt_modules value: {num_adapt_modules},"
                         f"and data type: {type(num_adapt_modules)}")

    if type(search_range) is not int or search_range < 1 or search_range > 10:
        raise ValueError("search_range needs to be an integer from 1-10."
                         f"\nDetected search_range value: {search_range},"
                         f"and data type: {type(search_range)}")

    # left and right image inputs are set to the same resolution
    left_input = layers.Input(shape=input_shape, name="left_input")
    right_input = layers.Input(shape=input_shape, name="right_input")

    # Initializing the layers
    layer_kwargs = {
        "kernel_size": (3, 3),
        "padding": "same",
        "activation": keras.layers.Activation(keras.activations.leaky_relu, dtype="float32", name="leaky_relu"),
        "use_bias": True
    }
    # Image feature pyramid (feature extractor)
    # F1
    conv1 = keras.layers.Conv2D(
        filters=16,
        strides=2,
        name="conv1",
        input_shape=(input_shape[0], input_shape[1], input_shape[2], ),
        **layer_kwargs)
    conv2 = keras.layers.Conv2D(filters=16, strides=1, name="conv2", **layer_kwargs)
    # F2
    conv3 = keras.layers.Conv2D(filters=32, strides=2, name="conv3", **layer_kwargs)
    conv4 = keras.layers.Conv2D(filters=32, strides=1, name="conv4", **layer_kwargs)
    # F3
    conv5 = keras.layers.Conv2D(filters=64, strides=2, name="conv5", **layer_kwargs)
    conv6 = keras.layers.Conv2D(filters=64, strides=1, name="conv6", **layer_kwargs)
    # F4
    conv7 = keras.layers.Conv2D(filters=96, strides=2, name="conv7", **layer_kwargs)
    conv8 = keras.layers.Conv2D(filters=96, strides=1, name="conv8", **layer_kwargs)
    # F5
    conv9 = keras.layers.Conv2D(filters=128, strides=2, name="conv9", **layer_kwargs)
    conv10 = keras.layers.Conv2D(filters=128, strides=1, name="conv10", **layer_kwargs)
    # F6
    conv11 = keras.layers.Conv2D(filters=192, strides=2, name="conv11", **layer_kwargs)
    conv12 = keras.layers.Conv2D(filters=192, strides=1, name="conv12", **layer_kwargs)

    #############################SCALE 6#################################
    M6 = ModuleM(layer="6", search_range=search_range)
    ############################SCALE 5###################################
    M5 = ModuleM(layer="5", search_range=search_range)
    ############################SCALE 4###################################
    M4 = ModuleM(layer="4", search_range=search_range)
    ############################SCALE 3###################################
    M3 = ModuleM(layer="3", search_range=search_range)
    ############################SCALE 2###################################
    M2 = ModuleM(layer="2", search_range=search_range)

    #######################PYRAMID FEATURES###############################
    # Left image feature pyramid (feature extractor)
    # F1
    left_pyramid = conv1(left_input)
    left_F1 = conv2(left_pyramid)
    # F2
    left_pyramid = conv3(left_F1)
    left_F2 = conv4(left_pyramid)
    # F3
    left_pyramid = conv5(left_F2)
    left_F3 = conv6(left_pyramid)
    # F4
    left_pyramid = conv7(left_F3)
    left_F4 = conv8(left_pyramid)
    # F5
    left_pyramid = conv9(left_F4)
    left_F5 = conv10(left_pyramid)
    # F6
    left_pyramid = conv11(left_F5)
    left_F6 = conv12(left_pyramid)

    # Right image feature pyramid (feature extractor)
    # F1
    right_pyramid = conv1(right_input)
    right_F1 = conv2(right_pyramid)
    # F2
    right_pyramid = conv3(right_F1)
    right_F2 = conv4(right_pyramid)
    # F3
    right_pyramid = conv5(right_F2)
    right_F3 = conv6(right_pyramid)
    # F4
    right_pyramid = conv7(right_F3)
    right_F4 = conv8(right_pyramid)
    # F5
    right_pyramid = conv9(right_F4)
    right_F5 = conv10(right_pyramid)
    # F6
    right_pyramid = conv11(right_F5)
    right_F6 = conv12(right_pyramid)

    #############################SCALE 6#################################
    D6 = M6([left_F6, right_F6])
    ############################SCALE 5###################################
    D5 = M5([left_F5, right_F5, D6])
    ############################SCALE 4###################################
    D4 = M4([left_F4, right_F4, D5])
    ############################SCALE 3###################################
    D3 = M3([left_F3, right_F3, D4])
    ############################SCALE 2###################################
    D2 = M2([left_F2, right_F2, D3])
    ############################REFINEMENT################################
    final_disparity = RefinementBlock(output_shape=input_shape)(left_F2, D2)

    # Monkey patch the train_step to use custom training
    keras.Model.train_step = _custom_train_step
    # Only need to monkey patch the predict_step if doing adaptation
    if num_adapt_modules != 0:
        keras.Model.last_adapt = keras.Variable(6)
        keras.Model.first_adapt_pass = True
        keras.Model.predict_step = _custom_predict_step(num_adapt_modules, mad_mode)
    keras.Model.test_step = _custom_test_step(keras.Model.predict_step)

    model = keras.Model(inputs={
                                "left_input": left_input,
                                "right_input": right_input
                            },
                           outputs=final_disparity,
                           name="MADNet")

    if weights in pretrained_weights:
        pretrained_models_url = "https://huggingface.co/ChristianOrr/madnet_keras/resolve/main/"
        model_name = "madnet_" + weights + ".h5"
        weight_path = pretrained_models_url + weights + ".h5"
        weights_path = get_file(model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
