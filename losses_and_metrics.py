import keras


#---------------Metrics-------------------
class EndPointError(keras.metrics.Metric):
    """
    End point error metric.
    Calculates the average absolute difference 
    between pixels in predicted disparity 
    and groundtruth.
    
    """
    def __init__(self, name="EPE", **kwargs):
        super(EndPointError, self).__init__(name=name, **kwargs)
        self.end_point_error = self.add_weight(name='EPE', initializer='zeros')
        self.total_steps = self.add_weight(name='total_steps', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        abs_errors = keras.ops.absolute(y_pred - y_true)
        # Valid map has all non-zero pixels set to 1 and 0 pixels remain 0
        valid_map = keras.ops.where(
            keras.ops.equal(y_true, 0),
            keras.ops.zeros_like(y_true, dtype="float32"),
            keras.ops.ones_like(y_true, dtype="float32")
        )
        # Remove the errors with 0 groundtruth disparity
        filtered_error = abs_errors * valid_map
        # Get the mean error (non-zero groundtruth pixels)
        self.end_point_error.assign_add(
            keras.ops.sum(filtered_error) / keras.ops.sum(valid_map)
        )
        self.total_steps.assign_add(1.0)

    def result(self):
        return self.end_point_error / self.total_steps

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.end_point_error.assign(0.0)
        self.total_steps.assign(0.0)


class Bad3(keras.metrics.Metric):
    """
    Bad3 also called D1-all is the percentage
    of pixels with disparity difference >= 3
    between predicted disparity and groundtruth.
    
    """
    def __init__(self, name="Bad3(%)", **kwargs):
        super(Bad3, self).__init__(name=name, **kwargs)
        self.pixel_threshold = 3
        self.bad3 = self.add_weight(name='bad3_percent', initializer='zeros')
        self.total_steps = self.add_weight(name='total_steps', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        abs_errors = keras.ops.absolute(y_pred - y_true)
        # Valid map has all non-zero pixels set to 1 and 0 pixels remain 0
        valid_map = keras.ops.where(
            keras.ops.equal(y_true, 0),
            keras.ops.zeros_like(y_true, dtype="float32"),
            keras.ops.ones_like(y_true, dtype="float32")
        )
        # Remove the errors with 0 groundtruth disparity
        filtered_error = abs_errors * valid_map
        # 1 assigned to all errors greater than threshold, 0 to the rest
        bad_pixel_abs = keras.ops.where(
            keras.ops.greater(filtered_error, self.pixel_threshold),
            keras.ops.ones_like(filtered_error, dtype="float32"),
            keras.ops.zeros_like(filtered_error, dtype="float32")
        )
        # (number of errors greater than threshold) / (number of errors)   
        self.bad3.assign_add((keras.ops.sum(bad_pixel_abs) / keras.ops.sum(valid_map)) * 100)
        self.total_steps.assign_add(1.0)

    def result(self):
        return self.bad3 / self.total_steps

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.bad3.assign(0.0)
        self.total_steps.assign(0.0)


#---------------Losses-------------------
class SSIMLoss(keras.losses.Loss):
    """
    SSIM dissimilarity measure
    Used for self-supervised training
    Args:
        y_true: target image
        y_pred: predicted image
    """
    def __init__(self, name="mean_SSIM_l1"):
        super(SSIMLoss, self).__init__(name=name)
        self.pool = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')
        self.reduction = None

    def call(self, y_true, y_pred):
        C1 = 0.01**2
        C2 = 0.03**2
        mu_x = self.pool(y_true)
        mu_y = self.pool(y_pred)

        sigma_x = self.pool(y_true**2) - mu_x**2
        sigma_y = self.pool(y_pred**2) - mu_y**2
        sigma_xy = self.pool(y_true*y_pred) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d
        SSIM = keras.ops.clip((1-SSIM)/2, 0, 1)

        mean_SSIM = keras.ops.mean(SSIM)

        sum_l1 = keras.ops.sum(keras.ops.absolute(y_true - y_pred))

        return 0.85 * mean_SSIM + 0.15 * sum_l1


class ReconstructionLoss(keras.losses.Loss):
    """
    Reconstruction loss function (sum l1)
    Per pixel absolute error between groundtruth 
    disparity and predicted disparity
    Used for supervised training
    Args:
        y_true: target images
        y_pred: predicted image
    """
    def __init__(self, name="sum_l1"):
        super(ReconstructionLoss, self).__init__(name=name)
        self.reduction = None

    def call(self, y_true, y_pred):
        # Valid map has all non-zero pixels set to 1 and 0 pixels remain 0
        valid_map = keras.ops.where(
            keras.ops.equal(y_true, 0),
            keras.ops.zeros_like(y_true, dtype="float32"),
            keras.ops.ones_like(y_true, dtype="float32")
        )
        return keras.ops.sum(valid_map * keras.ops.absolute(y_true-y_pred))
