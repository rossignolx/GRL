from k_gnn import ThreeMalkin, ThreeGlobal, TwoMalkin, ConnectedThreeMalkin

class TwoMalkinTransform:
    def __call__(self, data):
        data = TwoMalkin()(data)
        return data

class ConnectedThreeMalkinTransform:
    def __call__(self, data):
        data = ConnectedThreeMalkin()(data)
        return data
class ThreeMalkinTransform:
    def __call__(self, data):
        data = ThreeMalkin()(data)
        return data
class ThreeGlobalTransform:
    def __call__(self, data):
        data = ThreeGlobal()(data)
        return data

def get_transform(method):
    if method == "TwoMalkin":
        return TwoMalkinTransform()
    if method == "ThreeMalkin":
        return ThreeMalkinTransform()
    if method == "ConnectedThreeMalkin":
        return ConnectedThreeMalkinTransform()
    if method == "ThreeGlobal":
        return ThreeGlobalTransform()
    raise ValueError()