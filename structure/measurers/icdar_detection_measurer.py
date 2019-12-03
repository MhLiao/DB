import os
import subprocess
import shutil

import numpy as np
import json

from concern import Logger, AverageMeter
from concern.config import Configurable


class ICDARDetectionMeasurer(Configurable):
    def __init__(self, **kwargs):
        self.visualized = False

    def measure(self, batch, output):
        pairs = []
        for i in range(len(batch[-1])):
            pairs.append((batch[-1][i], output[i][0]))
        return pairs

    def validate_measure(self, batch, output):
        return self.measure(batch, output), [int(self.visualized)]

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch[0].shape[0]).tolist()

    def gather_measure(self, name, raw_metrics, logger: Logger):
        save_dir = os.path.join(logger.log_dir, name)
        shutil.rmtree(save_dir, ignore_errors=True)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        log_file_path = os.path.join(save_dir, name + '.log')
        count = 0
        for batch_pairs in raw_metrics:
            for _filename, boxes in batch_pairs:
                boxes = np.array(boxes).reshape(-1, 8).astype(np.int32)
                filename = 'res_' + _filename.replace('.jpg', '.txt')
                with open(os.path.join(save_dir, filename), 'wt') as f:
                    if len(boxes) == 0:
                        f.write('')
                    for box in boxes:
                        f.write(','.join(map(str, box)) + '\n')
                count += 1

        self.packing(save_dir)
        try:
            raw_out = subprocess.check_output(['python assets/ic15_eval/script.py -m=' + name
                                               + ' -g=assets/ic15_eval/gt.zip -s=' +
                                               os.path.join(save_dir, 'submit.zip') +
                                               '|tee -a ' + log_file_path],
                                              timeout=30, shell=True)
        except subprocess.TimeoutExpired:
            return {}
        raw_out = raw_out.decode().replace('Calculated!', '')
        dict_out = json.loads(raw_out)
        return {k: AverageMeter().update(v, n=count) for k, v in dict_out.items()}

    def packing(self, save_dir):
        pack_name = 'submit.zip'
        os.system(
            'zip -r -j -q ' +
            os.path.join(save_dir, pack_name) + ' ' + save_dir + '/*.txt')
