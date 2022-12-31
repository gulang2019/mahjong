from mahjong.model import ModelManager
def test():
    manager = ModelManager('../model/checkpoint', True)
    result = manager.get_best_model(100)
    print (result)
    
def vis():
    manager = ModelManager('../model/checkpoint', True)
    model = manager.get_latest_model()
    print(model)

import os 

def test1():
    manager = ModelManager('../model/checkpoint', True)
    files = list(os.listdir('../model/checkpoint'))
    comparors = []
    comparors = files[:4]
    idx = 4
    while idx < len(files): 
        result = manager.compare_models(*comparors, n_episode=20)
        worst_score = 1e9
        print (idx, result)
        for i in result:
            if worst_score > result[i][1]:
                worst_score = result[i][1]
                worst_candidate = i 
        comparors[worst_candidate] = files[idx] 
        idx += 1
    result = manager.compare_models(*comparors, 50)
    print ('final result', result)

def test2():
    manager = ModelManager('../model/checkpoint', True)
    result = manager.compare_models('model_12.pt', 'model_61.pt', 'model_11.pt', 'model_14.pt', n_episode=100)
    print(result)
# test()
# test1()
test2()
