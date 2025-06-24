import clip

def CLIP(model_version: str = 'ViT-B/3', device = 'cuda'):
    model_path = ''
    # if model_version == 'ViT-L/14':
    #     model_path = '/project/wanzhijing/DeepCore-main/ViT-L-14.pt'
    model, preprocess = clip.load(model_version, device, jit=False)
    return model, preprocess