import argparse
import os
import torch
import numpy as np
from concern.config import Configurable, Config


def main():
    parser = argparse.ArgumentParser(description='Convert model to ONNX')
    parser.add_argument('exp', type=str)
    parser.add_argument('resume', type=str, help='Resume from checkpoint')
    parser.add_argument('output', type=str, help='Output ONNX path')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Demo(experiment, experiment_args, cmd=args).inference()


class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        self.structure = experiment.structure
        self.model_path = self.args['resume']
        self.output_path = self.args['output']

    def init_torch_tensor(self):
        # Use gpu or not
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        states = torch.load(path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def inference(self):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        model.eval()

        img = np.random.randint(0, 255, size=(960, 960, 3), dtype=np.uint8)
        img = img.astype(np.float32)
        img = (img / 255. - 0.5) / 0.5  # torch style norm
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
        with torch.no_grad():
            img = img.to(self.device)
            torch.onnx.export(model.model.module, img, self.output_path, input_names=['input'],
                              output_names=['output'], dynamic_axes=dynamic_axes, keep_initializers_as_inputs=False,
                              verbose=False, opset_version=12)


if __name__ == '__main__':
    main()
