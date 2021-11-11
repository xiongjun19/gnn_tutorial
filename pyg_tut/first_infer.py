# coding=utf8


" Data(x=[2708, 1433], edge_index=[2, 10556]"

import torch
from first_ex import GCN

class GcnInfer(object):
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model = self.model.to('cuda')
        self.model.eval()

    def export(self, onnx_path):
        dm_x = torch.randn([500, 1433], dtype=torch.float32).to('cuda')
        dm_edge = torch.randint(500, [2, 5000]).to('cuda')
        # model = torch.jit.script(self.model)
        model = self.model
        torch.onnx.export(
            model,
            (dm_x, dm_edge),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['node_feature', 'edges'],
            output_names=['output'],
            dynamic_axes={
                'node_feature': {0: 'node_num'},
                'edges': {1: 'edge_num'},
                'output': {0: 'node_num'},
                },
            use_external_data_format=False
        )


def main(model_path, onnx_path):
    infer = GcnInfer(model_path)
    infer.export(onnx_path)



if __name__ == '__main__':
    import sys
    m_path = sys.argv[1]
    o_path = sys.argv[2]
    main(m_path, o_path)
