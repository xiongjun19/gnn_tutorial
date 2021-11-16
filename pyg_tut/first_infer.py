# coding=utf8


import torch
from first_ex import GCN
import onnx
import onnxruntime


class GcnInfer(object):
    def __init__(self, model_path, onnx_path):
        self.model = torch.load(model_path)
        self.model = self.model.to('cuda')
        self.model.eval()
        self.ort_session = onnxruntime.InferenceSession(onnx_path)

    def infer(self):
        first_dim = 2708
        sec_dim = 10556
        dm_x = torch.randn([first_dim, 1433], dtype=torch.float32).to('cuda')
        dm_edge = torch.randint(first_dim, [2, sec_dim]).to('cuda')
        # model = torch.jit.script(self.model)
        res = self.model(dm_x, dm_edge)
        print(res)
        print("following is onnx result")
        ort_inputs = self.ort_session.get_inputs()
        input_arr = [dm_x, dm_edge]
        input_dict = dict([
            (ort_elem.name, elem.to('cpu').numpy())
            for ort_elem, elem in zip(ort_inputs, input_arr)])
        res2 = self.ort_session.run(None, input_dict)[0]
        print(res2)


def main(model_path, onnx_path):
    infer = GcnInfer(model_path, onnx_path)
    infer.infer()


if __name__ == '__main__':
    import sys
    m_path = sys.argv[1]
    o_path = sys.argv[2]
    main(m_path, o_path)
