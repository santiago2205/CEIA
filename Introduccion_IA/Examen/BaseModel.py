class BaseModel:
    def __init__(self):
        self.model = None

    def fit(self, x, y):
        raise NotImplemented

    def predict(self, x):
        raise NotImplemented
