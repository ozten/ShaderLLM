# evaluation_utils.py
import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch
import logging

logger = logging.getLogger(__name__)

def validate_shader_compilation(shader_code, shader_type="fragment"):
    """
    Validates GLSL shader compilation using glslViewer.
    
    Args:
        shader_code (str): The GLSL shader source code
        shader_type (str): Type of shader ("fragment" or "vertex")
    
    Returns:
        dict: {
            "valid": bool,
            "error_message": str or None,
            "warnings": list of str
        }
    """
    result = {
        "valid": False,
        "error_message": None,
        "warnings": []
    }
    
    try:
        # Create temporary shader file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.frag', delete=False) as temp_file:
            temp_file.write(shader_code)
            temp_file_path = temp_file.name
        
        # Validate using glslViewer with headless mode
        cmd = ['glslViewer', temp_file_path, '--headless', '--validate']
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10  # 10 second timeout
        )
        
        if process.returncode == 0:
            result["valid"] = True
            if process.stderr:
                # Parse warnings from stderr
                lines = process.stderr.strip().split('\n')
                result["warnings"] = [line for line in lines if 'warning' in line.lower()]
        else:
            result["valid"] = False
            result["error_message"] = process.stderr.strip() if process.stderr else "Unknown compilation error"
            
    except subprocess.TimeoutExpired:
        result["error_message"] = "Shader compilation timed out"
    except FileNotFoundError:
        result["error_message"] = "glslViewer not found. Please install glslViewer."
        logger.warning("glslViewer not found. Shader validation will be skipped.")
    except Exception as e:
        result["error_message"] = f"Error during validation: {str(e)}"
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass
    
    return result

def render_shader_to_image(shader_code, output_path, resolution=(224, 224), timeout=30):
    """
    Renders a shader to an image using glslViewer.
    
    Args:
        shader_code (str): GLSL shader source code
        output_path (str): Path to save the rendered image
        resolution (tuple): (width, height) of output image
        timeout (int): Timeout in seconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create temporary shader file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.frag', delete=False) as temp_file:
            temp_file.write(shader_code)
            temp_file_path = temp_file.name
        
        # Render using glslViewer
        cmd = [
            'glslViewer',
            temp_file_path,
            '-e', f'screenshot,{output_path}',
            '-e', 'quit',
            '--headless',
            '-s', f'{resolution[0]},{resolution[1]}'
        ]
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        success = process.returncode == 0 and os.path.exists(output_path)
        
        if success:
            # Resize to exact target resolution using sips (macOS)
            try:
                subprocess.run([
                    'sips',
                    '-z', str(resolution[1]), str(resolution[0]),
                    output_path,
                    '--out', output_path
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                logger.warning("Failed to resize image with sips")
        
        return success
        
    except subprocess.TimeoutExpired:
        logger.error(f"Shader rendering timed out after {timeout} seconds")
        return False
    except FileNotFoundError:
        logger.error("glslViewer not found. Cannot render shader.")
        return False
    except Exception as e:
        logger.error(f"Error rendering shader: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

def compute_ssim_similarity(image1_path, image2_path):
    """
    Computes SSIM (Structural Similarity Index) between two images.
    
    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
    
    Returns:
        float: SSIM score between 0 and 1 (1 = identical)
    """
    try:
        # Load images
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        
        # Ensure same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        
        # Compute SSIM for each channel and average
        ssim_scores = []
        for channel in range(3):  # RGB
            score = ssim(
                img1_array[:, :, channel],
                img2_array[:, :, channel],
                data_range=255
            )
            ssim_scores.append(score)
        
        return np.mean(ssim_scores)
        
    except Exception as e:
        logger.error(f"Error computing SSIM: {str(e)}")
        return 0.0

def evaluate_generated_shader(shader_code, target_image_path, temp_dir=None):
    """
    Comprehensive evaluation of a generated shader.
    
    Args:
        shader_code (str): Generated GLSL shader code
        target_image_path (str): Path to target image for comparison
        temp_dir (str): Directory for temporary files
    
    Returns:
        dict: {
            "compilation": dict,  # Result from validate_shader_compilation
            "visual_similarity": float,  # SSIM score (0-1)
            "rendered_image_path": str or None
        }
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    results = {
        "compilation": validate_shader_compilation(shader_code),
        "visual_similarity": 0.0,
        "rendered_image_path": None
    }
    
    # Only attempt rendering if compilation succeeds
    if results["compilation"]["valid"]:
        # Generate unique filename for rendered image
        import uuid
        rendered_image_path = os.path.join(temp_dir, f"rendered_{uuid.uuid4().hex[:8]}.png")
        
        if render_shader_to_image(shader_code, rendered_image_path):
            results["rendered_image_path"] = rendered_image_path
            results["visual_similarity"] = compute_ssim_similarity(
                target_image_path, 
                rendered_image_path
            )
        else:
            logger.warning("Failed to render shader for visual similarity evaluation")
    
    return results

def batch_evaluate_shaders(shader_codes, target_image_paths, temp_dir=None):
    """
    Batch evaluation of multiple shader codes.
    
    Args:
        shader_codes (list): List of shader code strings
        target_image_paths (list): List of target image paths
        temp_dir (str): Directory for temporary files
    
    Returns:
        dict: {
            "compilation_rate": float,  # Percentage of shaders that compile
            "average_ssim": float,      # Average SSIM score
            "individual_results": list  # List of individual evaluation results
        }
    """
    if len(shader_codes) != len(target_image_paths):
        raise ValueError("Number of shader codes must match number of target images")
    
    individual_results = []
    compilation_successes = 0
    ssim_scores = []
    
    for shader_code, target_path in zip(shader_codes, target_image_paths):
        result = evaluate_generated_shader(shader_code, target_path, temp_dir)
        individual_results.append(result)
        
        if result["compilation"]["valid"]:
            compilation_successes += 1
            ssim_scores.append(result["visual_similarity"])
    
    compilation_rate = compilation_successes / len(shader_codes) if shader_codes else 0.0
    average_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    
    return {
        "compilation_rate": compilation_rate,
        "average_ssim": average_ssim,
        "individual_results": individual_results
    }