
from flask import Flask, request, jsonify, send_file, session
import requests
import base64
import io
import os
from PIL import Image
from dotenv import load_dotenv
import time
import torch
from diffusers import StableDiffusionInpaintPipeline

# Set up base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp_images')

# Create temp directory if it doesn't exist
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Load environment variables from .env file
load_dotenv(os.path.join(BASE_DIR, '.env'))

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')  # Gets secret key from .env
load_dotenv()




# Constants
STABLE_DIFFUSION_URL = "https://998cc0affd0b9f5651.gradio.live"
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Verify environment setup
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN not found in .env file")

# Headers for Replicate API
REPLICATE_HEADERS = {
    'Authorization': f'Token {REPLICATE_API_TOKEN}',
    'Content-Type': 'application/json',
}





def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.route('/edit', methods=['POST'])
def edit():
    try:
        data = request.json
        
        # Create unique filenames using timestamp
        timestamp = int(time.time())
        image_path = os.path.join(TEMP_DIR, f'temp_image_{timestamp}.png')
        mask_path = os.path.join(TEMP_DIR, f'temp_mask_{timestamp}.png')
        
        # Convert base64 image data to files
        image_data = data['image'].split(',')[1]
        mask_data = data['mask'].split(',')[1]
        
        # Save original image
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        
        # Process and save mask
        mask_image = Image.open(io.BytesIO(base64.b64decode(mask_data)))
        mask_image = mask_image.convert('L')
        mask_image = Image.eval(mask_image, lambda x: 255 - x)
        mask_image.save(mask_path)

        print("Files saved successfully")  # Debugging log


        # Prepare data for Replicate API
        payload = {
            "version": "c11bac58203367db93a3c552bd49a25a5418458ddcadf2c1fad4707d812149bf",
            "input": {
                "image": encode_image_to_base64(image_path),
                "mask": encode_image_to_base64(mask_path),
                "prompt": data['prompt'],
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "num_inference_steps": 30
            }
        }

        print("Making API request to Replicate")  # Debugging log

        # Call Replicate API
        response = requests.post(
            'https://api.replicate.com/v1/predictions',
            headers=REPLICATE_HEADERS,
            json=payload
        )
        response.raise_for_status()
        prediction = response.json()

        print(f"Initial prediction status: {prediction['status']}")  # Debugging log

        # Poll for result
        max_attempts = 30
        attempt = 0
        while prediction['status'] != 'succeeded' and attempt < max_attempts:
            time.sleep(1)  # Wait 1 second between checks
            response = requests.get(
                prediction['urls']['get'],
                headers=REPLICATE_HEADERS
            )
            prediction = response.json()
            print(f"Polling attempt {attempt}: {prediction['status']}")  # Debugging log
            
            if prediction['status'] == 'failed':
                raise Exception(f"Image editing failed: {prediction.get('error', 'Unknown error')}")
            
            attempt += 1

        if prediction['status'] != 'succeeded':
            raise Exception("Timeout waiting for image processing")

        # Get the edited image
        edited_image_url = prediction['output'][0]
        edited_image_response = requests.get(edited_image_url)
        edited_image_response.raise_for_status()

        print("Successfully received edited image")  # Debugging log

        # Clean up temporary files
        try:
            os.remove(image_path)
            os.remove(mask_path)
        except Exception as e:
            print(f"Error cleaning up files: {e}")  # Non-critical error

        # Return the edited image
        return send_file(
            io.BytesIO(edited_image_response.content),
            mimetype='image/png'
        )

    except Exception as e:
        print(f"Error in edit route: {str(e)}")  # Detailed error logging
        # Clean up files in case of error
        try:
            if 'image_path' in locals():
                os.remove(image_path)
            if 'mask_path' in locals():
                os.remove(mask_path)
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        
        return jsonify({"error": str(e)}), 500
        


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    
    payload = {
        "prompt": data['prompt'],
        "steps": data.get('steps', 30),
        "cfg_scale": data.get('cfg_scale', 7.5),
        "width": data.get('width', 512),
        "height": data.get('height', 512),
        "negative_prompt": "blurry, bad quality, distorted, disfigured, poor details, bad anatomy"
    }

    try:
        response = requests.post(f'{STABLE_DIFFUSION_URL}/sdapi/v1/txt2img', json=payload)
        response.raise_for_status()
        r = response.json()
        
        image_data = base64.b64decode(r['images'][0])
        image_stream = io.BytesIO(image_data)
        image_stream.seek(0)
        
        return send_file(
            image_stream,
            mimetype='image/png',
            as_attachment=False
        )
    
    except requests.exceptions.RequestException as e:
        print(f"Error in generate route: {str(e)}")  # Error logging
        return jsonify({"error": str(e)}), 500









@app.route('/')
def home():
    # (Previous HTML code remains the same)
    # I'm not including it to keep the response focused on the fixes
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Generator & Editor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
                align-items: center;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                background-color: #eee;
                border: none;
                border-radius: 5px;
            }
            .tab.active {
                background-color: #007bff;
                color: white;
            }
            .panel {
                display: none;
                width: 100%;
            }
            .panel.active {
                display: block;
            }
            .input-group {
                width: 100%;
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
            }
            .controls {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                width: 100%;
                margin: 20px 0;
            }
            .control-item {
                display: flex;
                flex-direction: column;
            }
            input[type="text"] {
                flex-grow: 1;
                padding: 10px;
                font-size: 16px;
            }
            input[type="number"], select {
                padding: 8px;
            }
            button {
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                border: none;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #0056b3;
            }
            #result, #editCanvas {
                max-width: 100%;
                margin-top: 20px;
                border: 2px solid #ddd;
            }
            .loading {
                display: none;
                margin: 20px 0;
            }
            .canvas-container {
                position: relative;
                margin-top: 20px;
            }
            .edit-controls {
                margin-top: 20px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Image Generator & Editor</h1>
            
            <div class="tabs">
                <button class="tab active" onclick="showPanel('generate')">Generate</button>
                <button class="tab" onclick="showPanel('edit')">Edit</button>
            </div>

            <!-- Generate Panel -->
            <div id="generatePanel" class="panel active">
                <div class="input-group">
                    <input type="text" id="prompt" placeholder="Enter your prompt...">
                </div>
                <div class="controls">
                    <div class="control-item">
                        <label for="steps">Steps:</label>
                        <input type="number" id="steps" value="30" min="20" max="150">
                    </div>
                    <div class="control-item">
                        <label for="cfg_scale">CFG Scale:</label>
                        <input type="number" id="cfg_scale" value="7.5" min="1" max="20" step="0.5">
                    </div>
                    <div class="control-item">
                        <label for="width">Width:</label>
                        <input type="number" id="width" value="512" min="256" max="1024" step="64">
                    </div>
                    <div class="control-item">
                        <label for="height">Height:</label>
                        <input type="number" id="height" value="512" min="256" max="1024" step="64">
                    </div>
                </div>
                <button onclick="generateImage()">Generate</button>
                <div class="loading" id="loading">Generating image...</div>
                <img id="result" style="display: none;">
            </div>

            <!-- Edit Panel -->
            <div id="editPanel" class="panel">
                <div class="input-group">
                    <input type="text" id="editPrompt" placeholder="Enter edit prompt...">
                </div>
                <div class="canvas-container">
                    <canvas id="editCanvas"></canvas>
                    <div class="edit-controls">
                        <div class="brush-controls">
                            <label for="brushSize">Brush Size:</label>
                            <input type="range" id="brushSize" min="5" max="50" value="20">
                            <span id="brushSizeValue">20</span>
                        </div>
                        <button onclick="clearMask()">Clear Mask</button>
                        <button onclick="applyEdit()">Apply Edit</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentImage = null;
            let isDrawing = false;
            let canvas, ctx;

            function showPanel(panelName) {
                document.querySelectorAll('.panel').forEach(panel => panel.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                
                document.getElementById(panelName + 'Panel').classList.add('active');
                event.target.classList.add('active');
                
                if (panelName === 'edit' && currentImage) {
                    setupEditCanvas();
                }
            }

            async function generateImage() {
                const prompt = document.getElementById('prompt').value;
                const steps = parseInt(document.getElementById('steps').value);
                const cfg_scale = parseFloat(document.getElementById('cfg_scale').value);
                const width = parseInt(document.getElementById('width').value);
                const height = parseInt(document.getElementById('height').value);
                
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                if (!prompt) {
                    alert('Please enter a prompt');
                    return;
                }

                loading.style.display = 'block';
                result.style.display = 'none';

                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            prompt: prompt,
                            steps: steps,
                            cfg_scale: cfg_scale,
                            width: width,
                            height: height,
                        }),
                    });

                    if (!response.ok) {
                        throw new Error('Generation failed');
                    }

                    const blob = await response.blob();
                    currentImage = URL.createObjectURL(blob);
                    result.src = currentImage;
                    result.style.display = 'block';
                } catch (error) {
                    alert('Error generating image: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                }
            }

            function setupEditCanvas() {
                canvas = document.getElementById('editCanvas');
                ctx = canvas.getContext('2d');
                
                const img = new Image();
                img.onload = function() {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    setupDrawing();
                };
                img.src = currentImage;
            }

            function setupDrawing() {
                canvas.addEventListener('mousedown', startDrawing);
                canvas.addEventListener('mousemove', draw);
                canvas.addEventListener('mouseup', stopDrawing);
                canvas.addEventListener('mouseout', stopDrawing);
                
                document.getElementById('brushSize').addEventListener('input', function(e) {
                    document.getElementById('brushSizeValue').textContent = e.target.value;
                });
            }

            function startDrawing(e) {
                isDrawing = true;
                draw(e);
            }

            function draw(e) {
                if (!isDrawing) return;
                
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const brushSize = parseInt(document.getElementById('brushSize').value);
                
                ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
                ctx.beginPath();
                ctx.arc(x, y, brushSize, 0, Math.PI * 2);
                ctx.fill();
            }

            function stopDrawing() {
                isDrawing = false;
            }

            function clearMask() {
                setupEditCanvas();
            }

            async function applyEdit() {
                const editPrompt = document.getElementById('editPrompt').value;
                if (!editPrompt) {
                    alert('Please enter an edit prompt');
                    return;
                }

                const imageBlob = await fetch(currentImage).then(r => r.blob());
                console.log(imageBlob);
                const maskData = canvas.toDataURL('image/png');

                const formData = new FormData();
                formData.append('image', imageBlob);
                formData.append('mask', maskData);
                formData.append('prompt', editPrompt);

                try {
                    const response = await fetch('/edit', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('Edit failed');
                    }

                    const blob = await response.blob();
                    currentImage = URL.createObjectURL(blob);
                    setupEditCanvas();
                } catch (error) {
                    alert('Error applying edit: ' + error.message);
                }
            }
        </script>
    </body>
    </html>
    """
    





   

# The home route remains the same...

if __name__ == '__main__':
    app.run(debug=True)