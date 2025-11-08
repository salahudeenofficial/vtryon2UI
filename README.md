# Virtual Try-On Gradio App (vtryon2UI)

A Gradio-based web application for virtual try-on that processes person images with masking and sends requests to an external API.

## Features

- **Image Masking**: Automatically masks upper body or lower body clothing with green overlay
- **Resolution Preservation**: Maintains original image resolution in output
- **API Integration**: Sends POST requests to external try-on API with masked images and prompts
- **User-Friendly UI**: Clean Gradio interface for easy interaction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/salahudeenofficial/vtryon2UI.git
cd vtryon2UI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure StableVITON checkpoints are available:
   - `StableVITON/checkpoints/humanparsing/parsing_atr.onnx`
   - `StableVITON/checkpoints/humanparsing/parsing_lip.onnx`
   - `StableVITON/checkpoints/openpose/body_pose_model.pth`

## Usage

### Running the Gradio App

```bash
python app.py
```

The app will launch on `http://0.0.0.0:7860`

### Using the Mask Script Directly

```bash
# Preserve input resolution (default)
python mask.py --mask_type upper_body --imagepath image.jpg

# Use fixed resolution
python mask.py --mask_type upper_body --imagepath image.jpg --no_preserve_resolution

# Specify output path
python mask.py --mask_type lower_body --imagepath image.jpg --output masked_image.jpg
```

## API Integration

The app sends POST requests to your API endpoint with:
- `masked_person_image`: Processed person image (with green mask for upper_body/lower_body)
- `cloth_image`: Garment image
- `prompt`: Generated prompt based on mask type

### Prompt Generation

- **upper_body/lower_body**: "by using the green masked area from Picture 3 as a reference for position place the garment from Picture 2 on the person from Picture 1."
- **other**: "tryon the garment in the Picture 2 on the person in Picture 1 .Dont't change the style and appearence of the garment and keep the garment look identical."

## Mask Types

- **upper_body**: Masks upper body clothing (shirt, top, etc.)
- **lower_body**: Masks lower body clothing (pants, skirt, etc.)
- **other**: No masking, uses original image

## Resolution Preservation

By default, the `masked_image()` function preserves the input image resolution:
- Processing happens at 576×768 for model efficiency
- Mask is scaled back to original resolution
- Final output matches input dimensions

To disable resolution preservation:
```bash
python mask.py --mask_type upper_body --imagepath image.jpg --no_preserve_resolution
```

## Project Structure

```
vtryon2UI/
├── app.py              # Gradio web application
├── mask.py             # Image masking functionality
├── requirements.txt    # Python dependencies
├── BasicSR/            # BasicSR library for image processing
└── StableVITON/        # StableVITON preprocessors (OpenPose, Parsing)
```

## Dependencies

- PyTorch 2.0.1
- Gradio 3.44.4
- OpenCV
- Pillow
- NumPy
- requests

See `requirements.txt` for complete list.

## License

See individual component licenses in `BasicSR/LICENSE/` and `StableVITON/` directories.

