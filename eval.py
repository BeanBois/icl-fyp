

import torch 




CONFIGS = {
    'model-file-path' : "instant_policy.pth",
    "num-demos" : 5,

}

def get_demos(num_demos):
    demo = []
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(CONFIGS['model-file-path'])

    model.eval()

    # start eval

    num_demos_given = CONFIG['num-demos']
    provided_demos = []

    