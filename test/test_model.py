from mahjong.model import ModelManager
def test():
    manager = ModelManager('../model/checkpoint', True)
    result = manager.get_best_model(100)
    print (result)
    
def vis():
    manager = ModelManager('../model/checkpoint', True)
    model = manager.get_latest_model()
    print(model)

def test1():
    manager = ModelManager('../model/checkpoint', True)
    result = manager.compare_models('model_27.pt', 'model_27.pt', 'model_27.pt', 'model_28.pt', 10)
    print (result)
    
# test()
test1()