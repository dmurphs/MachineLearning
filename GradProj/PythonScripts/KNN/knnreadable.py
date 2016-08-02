class KNNReadable:
    '''Object that can be read by KNN algorithm'''
    def __init__(self,measurements,name):
        self.measurements = measurements
        self.name = name
        self.guess = None
