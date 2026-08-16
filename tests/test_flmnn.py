from flmnn import FLMNN

def test_flmnn_initialization():
    model = FLMNN(
        Ms=(4, 4, 4),
        K=10,
        lambda_d1_grid=(1e-10,),
        lambda_d2_grid=(1e-10,),
        lambda_b_grid=(1e-8,),
        neighbors=(3,),
        mus=(0.5,),
        Ks=(5,),
    )

    assert model.K == 10
    assert model.Ms == (4, 4, 4)
