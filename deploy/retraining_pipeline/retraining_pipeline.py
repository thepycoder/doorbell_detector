class Pipeline:
    def __init__(self):
        pass

    def run(self):
        pass


class NumpyDataLoader:
    def __init__(self, path='/app/data'):
        self.path = path
    def load(self):
        ambient_path = os.path.join(self.path, 'ambient_data.npy')
        bell_path = os.path.join(self.path, 'bell_data.npy')

        ambient_data = np.load(open(ambient_path, 'rb'))
        bell_data = np.load(open(bell_path, 'rb'))


class MFCCPreProcessor:
    def __init__(self):
        pass