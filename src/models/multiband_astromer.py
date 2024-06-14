from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Layer,
    Dense,
    Concatenate,
)

from src.layers import (
    Encoder,
    RegLayer,
)
from src.layers.input import AddMSKToken


def build_input(
    window_size: int, batch_size: int | None = None, num_bands: int = 1
) -> dict[str, Input]:
    """
    Constructs input layers for each band, including layers for magnitudes, times, and
    attention masks.

    Parameters
    ----------
    window_size : int
        The size of the window for each input series.
    batch_size : int, optional
        The size of batches to be input into the model, by default None.
    num_bands : int
        The number of different bands to create inputs for.

    Returns
    -------
    dict[str, Input]
        A dictionary containing input layers for magnitudes, times, and attention masks for
        each band.
    """

    inputs = {}
    for i in range(num_bands):
        inputs[f'magnitudes_{i}'] = Input(
            shape=(window_size, 1),
            batch_size=batch_size,
            name=f'magnitudes_{i}',
        )

        inputs[f'times_{i}'] = Input(
            shape=(window_size, 1),
            batch_size=batch_size,
            name=f'times_{i}',
        )

        inputs[f'att_mask_{i}'] = Input(
            shape=(window_size, 1),
            batch_size=batch_size,
            name=f'att_mask_{i}',
        )

    return inputs


def get_multiband_ASTROMER(
    num_layers: int = 2,
    num_heads: int = 2,
    head_dim: int = 64,
    mixer_size: int = 256,
    dropout: float = 0.1,
    pe_base: int = 1000,
    pe_dim: int = 128,
    pe_c: int = 1,
    window_size: int = 100,
    batch_size: int | None = None,
    m_alpha: float = -0.5,
    mask_format: str = 'Q',
    use_leak: bool = False,
    loss_format: str = 'rmse',
    correct_loss: bool = False,
    temperature: float = 0.,
    num_bands: int = 1,
    dense_dim: int = 1024,
) -> Model:
    """
    Creates a multi-band ASTROMER model with separate encoders for each band. Each band's encoder
    processes its respective inputs independently.

    Parameters
    ----------
    num_layers : int
        Number of layers in each encoder.
    num_heads : int
        Number of attention heads in each encoder.
    head_dim : int
        Dimension of each attention head.
    mixer_size : int
        Size of the dense mixer layer for integrating encoder outputs.
    dropout : float
        Dropout rate applied in encoders.
    pe_base : int
        Base for positional encoding calculations.
    pe_dim : int
        Dimension of the positional encoding.
    pe_c : int
        Constant factor for positional encoding scaling.
    window_size : int
        Number of time steps each input series contains.
    batch_size : int, optional
        Batch size for training the model, by default None.
    m_alpha : float
        Alpha parameter for mask calculations.
    mask_format : str
        Format of the masking ('Q' for query, etc.).
    use_leak : bool
        Whether to use leaky components in the model.
    loss_format : str
        Format of the loss function used ('rmse', 'rmse+p', 'p').
    correct_loss : bool
        If true, corrections are applied to the loss calculation based on additional data.
    temperature : float
        Temperature parameter for softmax in attention calculations.
    num_bands : int
        Number of light bands in the dataset.
    dense_dim: int
        Dimensionality of the final dense layer.

    Returns
    -------
    Model
        A compiled TensorFlow model configured for multiple bands with independent encoders.
    """

    inputs = build_input(window_size, batch_size, num_bands)

    encoded_outputs = []
    for i in range(num_bands):
        band_input = {
            'input': inputs[f'magnitudes_{i}'],
            'times': inputs[f'times_{i}'],
            'mask_in': inputs[f'att_mask_{i}'],
        }

        msk_placeholder = AddMSKToken(
            trainable=True,
            window_size=window_size,
            on=[f'magnitudes_{i}'],
            name=f'msk_token_{i}',
        )(band_input)

        encoder = Encoder(
            window_size=window_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            mixer_size=mixer_size,
            dropout=dropout,
            pe_base=pe_base,
            pe_dim=pe_dim,
            pe_c=pe_c,
            m_alpha=m_alpha,
            mask_format=mask_format,
            use_leak=use_leak,
            temperature=temperature,
            name=f'encoder_{i}',
        )

        x = encoder(msk_placeholder)
        encoded_outputs.append(x)

    concatenated = Concatenate()(encoded_outputs) if num_bands > 1 else encoded_outputs[0]
    integrated = Dense(dense_dim, activation='relu', name='integration_layer')(concatenated)
    output = RegLayer(name='regression')(integrated)

    return CustomModel(
        correct_loss=correct_loss,
        loss_format=loss_format,
        inputs=list(inputs.values()),
        outputs=output,
        name='MultiBand_ASTROMER',
    )
