import h5py

try:
    # Open your model file in read/write mode
    with h5py.File('mlp.h5', 'r+') as f:
        config = f.attrs.get('model_config')
        
        if config is not None:
            # Decode the JSON string from the file
            config_str = config.decode('utf-8') if isinstance(config, bytes) else config
            
            # Find and replace the stubborn keyword
            if '"batch_shape"' in config_str:
                config_str = config_str.replace('"batch_shape"', '"batch_input_shape"')
                
                # Save it exactly how it was found
                f.attrs['model_config'] = config_str.encode('utf-8') if isinstance(config, bytes) else config_str
                print(" SUCCESS: 'mlp.h5' has been permanently patched! You can now run your Streamlit app.")
            else:
                print(" The file doesn't seem to contain 'batch_shape'. It might already be fixed.")
        else:
            print("Could not find the configuration inside the .h5 file.")
except Exception as e:
    print(f"An error occurred: {e}")