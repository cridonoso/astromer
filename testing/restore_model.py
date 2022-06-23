from core.astromer import get_ASTROMER


astromer = get_ASTROMER(num_layers=2,
                        d_model=256,
                        num_heads=4,
                        dff=128,
                        base=1000,
                        dropout=0.1,
                        maxlen=200,
                        use_leak=False,
                        no_train=False)

enc = astromer.get_layer('encoder')
for w in enc.enc_layers[0].ffn.variables:
    print(w.name, ' - ', w.shape)
