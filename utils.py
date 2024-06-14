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