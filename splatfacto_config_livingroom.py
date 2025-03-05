class SplatfactoModelConfig():
    """Splatfacto Model Config, partyly copied from Nerfstudio"""
    # visualize ref images
    vis: bool = False


    syn_data: bool = True
    # train
    stage_two_step: int = 3000
    stage_three_step: int = 4000
    stage_four_step: int = 20000

    # ransac
    py_ransac: bool = True
    ransac_threshold: float = 0.1

    # ablation study
    predict_depth: bool = True
    predict_normals: bool = True

    #loss
    depth_smooth_weight: float = 0.01
    normal_smooth_weight: float = 0.005
    predicted_normal_weight: float = 0.01
