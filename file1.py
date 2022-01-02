import os
class class_CPU:
    
    def __init__(self) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
    def inference(self,data, model_list):
        oup_results = {}
        """
            Your Entire Inference Code here
        """
        return oup_results
        
    