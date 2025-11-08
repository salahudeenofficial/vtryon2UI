#!/usr/bin/env python3
"""
Gradio app for virtual try-on with API integration.
Processes person images with masking and sends requests to external API.
"""

import os
import tempfile
from pathlib import Path
from PIL import Image
import requests
import gradio as gr
from mask import masked_image


def process_and_send(
    api_url: str,
    person_image: str,
    mask_type: str,
    cloth_image: str
) -> tuple:
    """
    Process person image, generate prompt, and send POST request to API.
    
    Args:
        api_url: API endpoint URL (e.g., "216.12.71.96:37577/tryon")
        person_image: Path to uploaded person image
        mask_type: One of 'upper_body', 'lower_body', or 'other'
        cloth_image: Path to uploaded cloth image
    
    Returns:
        tuple: (result_image_path, status_message)
    """
    # Validate inputs
    if not api_url or not api_url.strip():
        return None, "❌ Error: API URL is required"
    
    if not person_image:
        return None, "❌ Error: Person image is required"
    
    if not cloth_image:
        return None, "❌ Error: Cloth image is required"
    
    if mask_type not in ['upper_body', 'lower_body', 'other']:
        return None, f"❌ Error: Invalid mask_type: {mask_type}"
    
    try:
        # Normalize API URL (add http:// if not present)
        if not api_url.startswith(('http://', 'https://')):
            api_url = f"http://{api_url}"
        
        # Process person image based on mask_type
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_person:
            person_path = tmp_person.name
        
        try:
            if mask_type in ['upper_body', 'lower_body']:
                # Process person image with masking (preserve original resolution)
                masked_person_path = masked_image(
                    mask_type=mask_type,
                    imagepath=person_image,
                    output_path=person_path,
                    width=576,
                    height=768,
                    device_index=0,
                    preserve_resolution=True  # Preserve input image resolution
                )
                # Generate prompt for masked images
                prompt = "by using the green masked area from Picture 3 as a reference for position place the garment from Picture 2 on the person from Picture 1."
            else:  # mask_type == 'other'
                # For 'other', masked_image returns the original path, but we need a file we can read
                # So we'll copy the original to our temp file
                img = Image.open(person_image)
                img.convert('RGB').save(person_path, format='JPEG', quality=95)
                masked_person_path = person_path
                # Generate prompt for other type
                prompt = "tryon the garment in the Picture 2 on the person in Picture 1 .Dont't change the style and appearence of the garment and keep the garment look identical. "
                # prompt = "tryon the garment in the Picture 2 on the person in Picture 1 and then change the background"

            # Prepare files for API request
            with open(masked_person_path, 'rb') as f_person, open(cloth_image, 'rb') as f_cloth:
                files = {
                    'masked_person_image': ('masked_person.jpg', f_person, 'image/jpeg'),
                    'cloth_image': ('cloth.jpg', f_cloth, 'image/jpeg')
                }
                
                data = {
                    'prompt': prompt
                }
                
                # Send POST request to API
                response = requests.post(api_url, files=files, data=data, timeout=300)
                
                if response.status_code == 200:
                    # Save response image
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_result:
                        result_path = tmp_result.name
                    
                    # Check if response is an image
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        with open(result_path, 'wb') as f:
                            f.write(response.content)
                        return result_path, f"✅ Success! Image processed and sent to API.\nPrompt: {prompt}"
                    else:
                        # Try to parse as JSON or text
                        try:
                            result_json = response.json()
                            return None, f"✅ API Response (Status 200):\n{result_json}"
                        except:
                            return None, f"✅ API Response (Status 200):\n{response.text[:500]}"
                else:
                    error_msg = f"❌ API Error (Status {response.status_code}): {response.text[:500]}"
                    return None, error_msg
                    
        finally:
            # Clean up temporary person image
            if os.path.exists(person_path) and person_path != masked_person_path:
                try:
                    os.unlink(person_path)
                except:
                    pass
            
    except FileNotFoundError as e:
        return None, f"❌ Error: File not found - {str(e)}"
    except requests.exceptions.RequestException as e:
        return None, f"❌ Network Error: Failed to connect to API - {str(e)}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_interface():
    """Create and return Gradio interface."""
    
    with gr.Blocks(title="Virtual Try-On API Client", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🎨 Virtual Try-On API Client
            
            Upload a person image and cloth image, select mask type, and send to API for virtual try-on.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                api_url_input = gr.Textbox(
                    label="API URL",
                    placeholder="216.12.71.96:37577/tryon",
                    value="",
                    info="Enter the API endpoint URL (e.g., 216.12.71.96:37577/tryon)"
                )
                
                mask_type_dropdown = gr.Dropdown(
                    label="Mask Type",
                    choices=['upper_body', 'lower_body', 'other'],
                    value='upper_body',
                    info="Select the type of garment to try on"
                )
                
                person_image_input = gr.Image(
                    label="Person Image",
                    type="filepath",
                    sources=["upload"],
                    info="Upload the person image"
                )
                
                cloth_image_input = gr.Image(
                    label="Cloth Image",
                    type="filepath",
                    sources=["upload"],
                    info="Upload the garment image"
                )
                
                process_btn = gr.Button("🚀 Process & Send to API", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Result Image",
                    type="filepath"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False
                )
        
        # Process button click handler
        process_btn.click(
            fn=process_and_send,
            inputs=[api_url_input, person_image_input, mask_type_dropdown, cloth_image_input],
            outputs=[output_image, status_output]
        )
        
        gr.Markdown(
            """
            ### 📝 Instructions:
            1. Enter the API endpoint URL
            2. Select mask type:
               - **upper_body**: Masks upper body clothing (shirt, top, etc.)
               - **lower_body**: Masks lower body clothing (pants, skirt, etc.)
               - **other**: No masking, full body try-on
            3. Upload person image
            4. Upload cloth/garment image
            5. Click "Process & Send to API"
            
            ### ℹ️ Note:
            - For `upper_body` and `lower_body`, the person image will be processed with green masking
            - For `other`, the original person image will be used
            - The API will receive: masked_person_image, cloth_image, and prompt
            """
        )
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)

