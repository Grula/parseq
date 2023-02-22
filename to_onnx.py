import torch
from torch.autograd import Variable

from strhub.models.parseq.system import PARSeq

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

if __name__ == '__main__':
    parseq = torch.hub.load('baudm/parseq', 'trba', pretrained=True, refine_iters=0).eval()

    dummy_input = Variable(torch.randn(1, 3, *parseq.hparams.img_size))
    parseq.to_onnx('parseq.onnx', dummy_input, opset_version=16, 
        input_names = ['input'],
        dynamic_axes={
            'input' : {0 : 'batch', 2:'height', 3: 'width'},
        },) # opset v14 or newer is required
    # torch.onnx.export(parseq,
    #     dummy_input,
    #     "parseq.onnx",
    #     input_names = ['input'],
    #     dynamic_axes={
    #         'input' : {0 : 'batch', 2:'height', 3: 'width'},
    #     },
    #     opset_version= 16,
    #     )

