"""
Hugging Face Image Generator
Free alternative to Replicate for AI image generation using Hugging Face Inference API
"""

import os
import base64
import requests
import logging
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import time

logger = logging.getLogger(__name__)

class HuggingFaceImageGenerator:
    """
    Image generator using Hugging Face's free Inference API
    """
    
    def __init__(self, api_token=None, model="stabilityai/stable-diffusion-xl-base-1.0", max_tokens=500):
        """
        Initialize the Hugging Face image generator
        
        Args:
            api_token: Hugging Face API token (free)
            model: The Hugging Face model to use
            max_tokens: Maximum token limit for prompts
        """
        self.model = model
        self.max_tokens = max_tokens
        self.fallback_mode = False
        
        # Set up Hugging Face API token
        if api_token:
            self.api_token = api_token
        else:
            self.api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        
        if not self.api_token:
            logger.error("Hugging Face API token not found. Please set HUGGINGFACE_API_TOKEN or pass api_token parameter.")
            self.fallback_mode = True
        else:
            logger.info(f"Initialized Hugging Face image generator with model: {self.model}")
    
    @staticmethod
    def test_connection(api_token):
        """Test if the Hugging Face API connection works"""
        if not api_token:
            return {
                "success": False,
                "message": "No Hugging Face API token provided",
                "api_token_set": False
            }
        
        try:
            headers = {"Authorization": f"Bearer {api_token}"}
            # Test with a simple model info request
            response = requests.get(
                "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Successfully connected to Hugging Face API")
                return {
                    "success": True,
                    "message": "Hugging Face API connection successful",
                    "api_token_set": True
                }
            else:
                return {
                    "success": False,
                    "message": f"Hugging Face API returned status {response.status_code}",
                    "api_token_set": True
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to connect to Hugging Face API: {str(e)}",
                "api_token_set": True
            }
    
    def generate_image(self, prompt, width=512, height=512, steps=20, guidance=7.5):
        """
        Generate an image using Hugging Face Inference API
        
        Args:
            prompt: Text description of the image to generate
            width: Image width (default 512)
            height: Image height (default 512)
            steps: Number of inference steps (default 20)
            guidance: Guidance scale (default 7.5)
            
        Returns:
            Dictionary with image data and metadata
        """
        if self.fallback_mode:
            logger.warning("Hugging Face generator in fallback mode, creating text image")
            return self._create_fallback_image(prompt, width, height)
        
        try:
            # Clean and prepare the prompt
            cleaned_prompt = self.refine_prompt(prompt)
            
            logger.info(f"Generating image for prompt: '{cleaned_prompt[:50]}...'")
            logger.info(f"Image parameters: {width}x{height}, steps: {steps}, guidance: {guidance}")
            
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            # Use the Inference API endpoint
            api_url = f"https://api-inference.huggingface.co/models/{self.model}"
            
            payload = {
                "inputs": cleaned_prompt,
                "parameters": {
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance
                }
            }
            
            logger.info("Starting image generation with Hugging Face API...")
            logger.info(f"Calling Hugging Face API with model: {self.model}")
            
            start_time = time.time()
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=120  # Generous timeout for image generation
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                logger.info(f"Image generation completed in {end_time - start_time:.2f} seconds")
                
                # The response should contain the image data directly
                image_data = response.content
                
                # Convert to base64
                img_str = base64.b64encode(image_data).decode()
                
                logger.info(f"Successfully generated image for prompt: {cleaned_prompt}")
                
                return {
                    "image_base64": img_str,
                    "prompt": cleaned_prompt,
                    "fallback": False,
                    "image_url": None  # HF doesn't provide URLs
                }
            
            elif response.status_code == 503:
                # Model is loading, wait and retry
                logger.info("Model is loading, waiting 20 seconds and retrying...")
                time.sleep(20)
                
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    image_data = response.content
                    img_str = base64.b64encode(image_data).decode()
                    
                    return {
                        "image_base64": img_str,
                        "prompt": cleaned_prompt,
                        "fallback": False,
                        "image_url": None
                    }
                else:
                    raise Exception(f"Hugging Face API error after retry: {response.status_code} - {response.text}")
            
            else:
                raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error generating image with Hugging Face: {str(e)}")
            
            # Check if it's a quota/rate limit error
            error_str = str(e).lower()
            if "quota" in error_str or "rate limit" in error_str or "429" in error_str:
                error_message = "🕒 Free API Quota Exceeded!\n\nPlease wait a few minutes and try again.\nHugging Face free tier has usage limits."
            elif "503" in error_str or "loading" in error_str:
                error_message = "⏳ Model Loading...\n\nThe AI model is starting up.\nThis can take 1-2 minutes.\nPlease try again shortly."
            else:
                error_message = f"Hugging Face API error: {str(e)[:100]}"
            
            # Return fallback image with error message
            return self._create_fallback_image(f"⚠️ Image Generation Failed\n\n{error_message}\n\nPrompt: {prompt}", width, height)
    
    def _create_fallback_image(self, text, width=512, height=512):
        """Create a text-based fallback image"""
        try:
            image = create_text_image(text, width, height)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                "image_base64": img_str,
                "prompt": text,
                "fallback": True,
                "image_url": None
            }
        except Exception as img_error:
            logger.error(f"Error creating fallback image: {str(img_error)}")
            raise RuntimeError(f"Complete failure in image generation: {str(img_error)}")
    
    def refine_prompt(self, prompt):
        """
        Clean and optimize the prompt for Hugging Face models, with special handling for multiple subjects
        
        Args:
            prompt: The original prompt
            
        Returns:
            Cleaned prompt with better multi-subject handling
        """
        # Clean the prompt
        prompt = prompt.strip()
        
        # Remove common prefixes that users might add
        prefixes_to_remove = ["generate image of", "create image of", "make image of", "draw", "paint"]
        for prefix in prefixes_to_remove:
            if prompt.lower().startswith(prefix):
                prompt = prompt[len(prefix):].strip()
        
        # Remove quotes if the prompt is wrapped in them
        if prompt.startswith('"') and prompt.endswith('"'):
            prompt = prompt[1:-1]
        
        # Enhanced handling for multiple subjects
        # Look for "and" patterns that indicate multiple subjects
        if " and " in prompt.lower():
            # Common patterns like "cat and dog" 
            words = prompt.split()
            and_indices = [i for i, word in enumerate(words) if word.lower() == "and"]
            
            for and_index in and_indices:
                if and_index > 0 and and_index < len(words) - 1:
                    # Check if it's likely an animal/object pair
                    before = words[and_index - 1].lower()
                    after = words[and_index + 1].lower()
                    
                    # Common subjects that often get missed
                    animals = ["cat", "dog", "bird", "fish", "horse", "cow", "pig", "sheep", "rabbit", "mouse"]
                    objects = ["car", "house", "tree", "flower", "ball", "book", "chair", "table"]
                    
                    if before in animals + objects and after in animals + objects:
                        # Emphasize both subjects
                        prompt = prompt.replace(f"{words[and_index-1]} and {words[and_index+1]}", 
                                              f"both a {words[and_index-1]} and a {words[and_index+1]}")
                        logger.info(f"Enhanced multi-subject prompt: emphasized both subjects")
                        break
        
        # Apply multi-subject optimization
        prompt = self.optimize_for_multi_subjects(prompt)
        
        logger.info(f"Using cleaned prompt: {prompt}")
        return prompt
    
    def optimize_for_multi_subjects(self, prompt):
        """
        Special optimization for prompts with multiple subjects
        
        Args:
            prompt: The prompt to optimize
            
        Returns:
            Optimized prompt with better multi-subject keywords
        """
        # Keywords that help Stable Diffusion generate multiple subjects
        multi_subject_enhancers = [
            "together", "side by side", "both visible", "scene with", "group of", 
            "featuring both", "composition showing", "scene including"
        ]
        
        # If we detect multiple subjects, add helpful keywords
        if any(word in prompt.lower() for word in ["both a", "and a", "alongside"]):
            # Add composition keywords that help with multiple subjects
            if "beach" in prompt.lower():
                prompt += ", wide shot composition"
            else:
                prompt += ", group composition"
                
            # Add emphasis on both subjects being visible
            if not any(enhancer in prompt.lower() for enhancer in multi_subject_enhancers):
                prompt += ", both subjects clearly visible"
        
        return prompt

    def count_tokens(self, text):
        """
        Rough token count estimation
        """
        return len(text.split())


def create_text_image(text, width=512, height=512):
    """
    Create a simple text image as fallback
    
    Args:
        text: Text to display
        width: Image width
        height: Image height
        
    Returns:
        PIL Image object
    """
    # Create image with light background
    image = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font_size = max(12, min(width, height) // 25)
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    
    if font:
        # Word wrap the text
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font and _get_text_width(test_line, font) < width - 40:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate total text height
        line_height = font_size + 5
        total_text_height = len(lines) * line_height
        
        # Start drawing from center
        y_offset = (height - total_text_height) // 2
        
        for line in lines:
            text_width = _get_text_width(line, font)
            x_offset = (width - text_width) // 2
            draw.text((x_offset, y_offset), line, fill=(60, 60, 60), font=font)
            y_offset += line_height
    
    return image


def _get_text_width(text, font):
    """Get the width of text using the given font"""
    try:
        # For newer Pillow versions
        if hasattr(ImageDraw.Draw(Image.new('RGB', (1, 1))), 'textlength'):
            return ImageDraw.Draw(Image.new('RGB', (1, 1))).textlength(text, font=font)
        else:
            # For older Pillow versions
            return ImageDraw.Draw(Image.new('RGB', (1, 1))).textsize(text, font=font)[0]
    except Exception:
        # Fallback estimation
        return len(text) * 8
