from mahjong.model import ModelManager
def test():
    manager = ModelManager('../model/checkpoint', True)
    result = manager.get_best_model('model_1.pt', 'model_1.pt', 'model_1.pt', 'model_220.pt')
    print (result)
test()