import os


class SignalMonitor(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def get_signal(self):
        if self.file_path is None:
            return None
        if os.path.exists(self.file_path):
            with open(self.file_path) as f:
                data = f.read()
                os.remove(self.file_path)
                return data
        else:
            return None
