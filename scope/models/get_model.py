from scope.models.cnn3d import get_cnn3d_model
from scope.models.resnet3d import Resnet3DBuilder


def get_model(width: int, height: int, depth: int, n_channels: int, model_type: str = 'cnn3d',
              weight_decay_coefficient=1e-4, regression=False):
    if model_type == 'cnn3d':
        model = get_cnn3d_model(width=width, height=height, depth=depth,
                                channels=n_channels, weight_decay=weight_decay_coefficient, regression=regression)
    elif model_type == 'resnet3d':
        model = Resnet3DBuilder().build_resnet_50(
            (width, height, depth, n_channels),
            1, regression=regression, reg_factor=weight_decay_coefficient)
    elif model_type == 'efficientnet3d':
        import scope.models.efficientnet_3D.tfkeras as efn
        model = efn.EfficientNetB0(input_shape=(width, height, depth, n_channels),
                                   weights=None, classes=1, include_top=True, regression=regression)
    else:
        raise ValueError(f'{model_type} is not implemented yet.')
    return model
