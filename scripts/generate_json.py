import os
import json
from glob import glob

def create_json_configs(data_dir):
    # Find all .png files (excluding those with $ in name)
    png_files = [f for f in glob(f"{data_dir}/*.png") if '$' not in f]
    
    for png_file in png_files:
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        frag_file = os.path.join(data_dir, f"{base_name}.frag")
        
        # Skip if we don't have a corresponding fragment shader
        if not os.path.exists(frag_file):
            continue
            
        # Read the fragment shader code
        with open(frag_file, 'r') as f:
            shader_code = f.read()
        
        # Create the example JSON structure
        example = {
            "instruction": "Generate an HLSL shader that reproduces this image",
            "input": f"Image: {os.path.basename(png_file)}",
            "output": shader_code
        }
        
        # Write the JSON file
        json_path = os.path.join(data_dir, f"example_{base_name}.json")
        with open(json_path, 'w') as f:
            json.dump(example, f, indent=2)
            
        print(f"Created {json_path}")

# Process both train and val directories
create_json_configs("./data/train")
create_json_configs("./data/val")

print("JSON configuration files created successfully.")