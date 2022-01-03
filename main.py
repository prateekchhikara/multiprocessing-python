from file1 import class_CPU
from file2 import class_GPU
from multiprocessing import Process, Manager
import time

class Main:
    
    def __init__(self) -> None:
        self.manager = Manager()
        self.return_dict = self.manager.dict()
        self.initial_time = time.time()
       
    def CPU_module(self, data, model_list):
        """ This method performs inference using the given data-path on mentioned
        models.

        Args:
            data (str): this is the path of the dataset
            model_list (list): the list of CPU models that we want to inference on.
        """
        obj_cpu = class_CPU()
        output_res = obj_cpu.inference(data, model_list)
        self.return_dict['CPU_output'] = output_res
        print(f"total CPU time : {time.time() - self.initial_time}")
        
    def GPU_module(self, data, model_list):
        """ This method performs inference using the given data-path on mentioned
        models.

        Args:
            data (str): this is the path of the dataset
            model_list (list): the list of GPU models that we want to inference on.
        """
        obj_gpu = class_GPU()
        output_res = obj_gpu.inference(data, model_list)
        self.return_dict['GPU_output'] = output_res
        print(f"total GPU time : {time.time() - self.initial_time}")
        
    def run(self, data, cpu_model_list, gpu_model_list):
        """ This is the main code which performs multi-processing. 

        Args:
            data (str): this is the path of the dataset
            cpu_model_list (list): the list of CPU models that we want to inference on
            gpu_model_list (list): the list of GPU models that we want to inference on

        Returns:
            dict: result of the inference
        """
        # running CPU models
        process1 = Process(target=self.CPU_module, args=(data, cpu_model_list))
        process1.start()
        # running CPU models
        process2 = Process(target=self.GPU_module, args=(data, gpu_model_list))
        process2.start()
        # joining the process
        process1.join()
        process2.join()
        
        print(f"Total time : {time.time() - self.initial_time}")
        
        return self.return_dict
    

if __name__ == "__main__":
    obj = Main()
    dataset_path = "/home/prateek/...."
    cpu_model_list = ['...', '...']
    gpu_model_list = ['...', '...']
    predictions = obj.run(dataset_path, cpu_model_list, gpu_model_list)
        
    