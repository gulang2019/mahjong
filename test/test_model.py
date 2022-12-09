from mahjong.model import ModelManager
def test():
    manager = ModelManager('../model/checkpoint', True)
    result = manager.get_best_model()
    print (result)
test()