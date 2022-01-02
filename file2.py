import os
class class_GPU:
    
    def __init__(self) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
    def inference(self,data, model_list):
        oup_results = {}
        """
            Your Entire Inference Code here
        """
        return oup_results