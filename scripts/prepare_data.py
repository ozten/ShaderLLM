import os
import subprocess
from pathlib import Path

def process_frag_files(root_dir):
    # Find all .frag files recursively under root_dir
    frag_files = list(Path(root_dir).rglob("*.frag"))
    png_files = []
    
    for frag_file in frag_files:
        # Extract the number part (NNN)
        num = frag_file.stem
        
        # Get the directory path relative to data
        relative_dir = frag_file.parent.relative_to(root_dir)
        
        # Construct paths
        frag_path = str(frag_file)
        png_path = str(frag_file.with_suffix('.png'))
        png_files.append(png_path)  # Add to list for HTML generation
        
        # Skip if PNG already exists and is newer than frag file
        if os.path.exists(png_path) and os.path.getmtime(png_path) > os.path.getmtime(frag_path):
            print(f"Skipping {frag_path}, PNG already exists and is up-to-date")
            continue
        
        print(f"Processing {frag_path}...")
        
        # Step 1: Generate screenshot with glslViewer
        glsl_cmd = [
            'glslViewer',
            frag_path,
            '-e', f'screenshot,{png_path}',
            '-e', 'quit',
            '--headless'
        ]
        
        # Step 2: Resize with sips
        sips_cmd = [
            'sips',
            '-z', '224', '224',
            png_path,
            '--out', png_path
        ]
        
        try:
            # Run glslViewer
            subprocess.run(glsl_cmd, check=True)
            
            # Run sips
            subprocess.run(sips_cmd, check=True)
            
            print(f"Successfully created {png_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {frag_path}: {e}")
    
    return png_files

def generate_html(png_files, output_path='index.html'):
    # Sort files alphabetically
    png_files.sort()
    
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>GLSL Screenshots</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .image-container {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        img {{
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }}
        .filename {{
            margin-top: 5px;
            font-size: 14px;
            color: #666;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <h1>GLSL Screenshots</h1>
    <p>Total images: {len(png_files)}</p>
    <div class="gallery">
"""

    for png_file in png_files:
        html_content += f"""
        <div class="image-container">
            <img src="{png_file}" alt="{png_file}">
            <div class="filename">{png_file}</div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated {output_path} with {len(png_files)} images")

if __name__ == "__main__":
    # Process both train and val directories
    base_dir = 'data'
    all_png_files = []
    
    for subdir in ['train', 'val']:
        dir_path = os.path.join(base_dir, subdir)
        if os.path.exists(dir_path):
            png_files = process_frag_files(dir_path)
            all_png_files.extend(png_files)
        else:
            print(f"Directory {dir_path} does not exist, skipping")
    
    # Generate HTML index
    if all_png_files:
        generate_html(all_png_files)
    else:
        print("No PNG files found to generate HTML")