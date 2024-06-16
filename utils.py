def print_model_info(model):
    print(f'Model size : {model.get_memory_footprint() // 1024 ** 2} MB')

    try:
        print(f'Model is quantized : {model.is_quantized}')
        print(f'Model quantization method : {model.quantization_method}')
    except:
        print(f'Model is quantized : False')
        pass
        
    try:
        print(f'Model 8-bit quantized : {model.is_loaded_in_8bit}')
    except:
        pass
    
    try:
        print(f'Model 4-bit quantized : {model.is_loaded_in_4bit}')
    except:
        pass
    
    print(f'Model on GPU : {next(model.parameters()).is_cuda}')

def print_device_info():
    import pynvml
    
    pynvml.nvmlInit()
    
    print(f'driver version : {pynvml.nvmlSystemGetDriverVersion()}')
    
    devices = pynvml.nvmlDeviceGetCount()
    print(f'device count : {devices}')
    
    for i in range(devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        print(f'device {i} : {pynvml.nvmlDeviceGetName(handle)}')
    
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f'device {i} : mem total : {info.total // 1024 ** 2} MB')
        print(f'device {i} : mem used  : {info.used // 1024 ** 2} MB')
        print(f'device {i} : mem free  : {info.free // 1024 ** 2} MB')
