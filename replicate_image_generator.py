import os
import logging
import base64
import sys
import platform
from io import BytesIO
import requests
import replicate
from PIL import Image
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define fallback image generator function
def create_text_image(text, width=512, height=512):
    """Create a simple text-based image when image generation fails"""
    from PIL import Image, ImageDraw, ImageFont
    try:
        # Create a blank image with a gradient background
        image = Image.new('RGB', (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(image)
        
        # Add a simple gradient background
        for y in range(height):
            r = int(240 - (y / height) * 40)
            g = int(240 - (y / height) * 20)
            b = int(240 - (y / height) * 30)
            for x in range(width):
                draw.point((x, y), fill=(r, g, b))
        
        # Try to use a nice font, fall back to default if not available
        try:
            font_size = 24
            if platform.system() == "Windows":
                font = ImageFont.truetype("arial.ttf", font_size)
            else:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Split text into lines
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if draw.textlength(test_line, font=font) < width - 40:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Draw text centered on image
        y_position = height // 2 - (len(lines) * font_size) // 2
        for line in lines:
            text_width = draw.textlength(line, font=font)
            x_position = (width - text_width) // 2
            draw.text((x_position, y_position), line, font=font, fill=(0, 0, 0))
            y_position += font_size + 10
        
        # Add a border
        draw.rectangle([(10, 10), (width-10, height-10)], outline=(200, 200, 200), width=2)
        
        return image
    except Exception as e:
        # Last resort - create an ultra-simple image
        logger.error(f"Error creating text image: {str(e)}")
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        return img

class ReplicateImageGenerator:
    def __init__(self, api_token=None, model="stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478", max_tokens=200):
        """
        Initialize the Replicate image generator.
        
        Args:
            api_token: Replicate API token (if not provided, will check environment)
            model: The Replicate model to use for image generation
            max_tokens: Maximum token limit for prompts
        """
        self.model = model
        self.max_tokens = max_tokens
        self.fallback_mode = False
        
        # Set up Replicate API token
        if api_token:
            self.api_token = api_token
            os.environ["REPLICATE_API_TOKEN"] = api_token
        else:
            self.api_token = os.environ.get("REPLICATE_API_TOKEN")
        
        if not self.api_token:
            logger.error("Replicate API token not found. Please set REPLICATE_API_TOKEN environment variable or pass api_token parameter.")
            self.fallback_mode = True
        else:
            logger.info(f"Initialized Replicate image generator with model: {model}")
            
        # Test the connection
        try:
            if not self.fallback_mode:
                # Test connection by getting the model
                replicate.models.get(model.split("/")[0] + "/" + model.split("/")[1].split(":")[0])
                logger.info("Successfully connected to Replicate API")
        except Exception as e:
            logger.warning(f"Could not verify Replicate connection: {str(e)}")
            logger.warning("Will attempt generation anyway")
    
    def generate_image(self, prompt, negative_prompt=None, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, max_tokens=None):
        """
        Generate an image using Replicate API.
        
        Args:
            prompt: The text prompt to generate an image from
            negative_prompt: What not to include in the image
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            max_tokens: Maximum number of tokens allowed (uses instance default if None)
            
        Returns:
            A dictionary containing:
            - image_base64: Base64 encoded image
            - prompt: The original prompt
            - image_url: URL to the generated image (from Replicate)
        """
        # Use instance default if not provided
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        logger.info(f"Generating image for prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        logger.info(f"Image parameters: {width}x{height}, steps: {num_inference_steps}, guidance: {guidance_scale}")
        
        # Check and optimize prompt if needed
        original_prompt = prompt
        if self.count_tokens(prompt) > max_tokens:
            prompt = self.smart_prompt_optimization(prompt, max_tokens)
            if self.count_tokens(prompt) > max_tokens:
                prompt = self.truncate_prompt(prompt, max_tokens)
        
        # If we're in fallback mode, just generate a text image
        if self.fallback_mode:
            logger.warning("Using fallback text-to-image mode (no Replicate API token)")
            try:
                image = create_text_image(f"Replicate API not configured. Original prompt: {prompt}", width, height)
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return {
                    "image_base64": img_str,
                    "prompt": prompt,
                    "fallback": True,
                    "image_url": None
                }
            except Exception as e:
                logger.error(f"Error creating fallback image: {str(e)}")
                raise RuntimeError(f"Failed to generate image: {str(e)}")

        # Normal generation path using Replicate API
        try:
            logger.info("Starting image generation with Replicate API...")
            
            # Prepare the input for the model
            input_params = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": "K_EULER"
            }
            
            # Add negative prompt if provided
            if negative_prompt:
                input_params["negative_prompt"] = negative_prompt
                
            logger.info(f"Calling Replicate API with model: {self.model}")
            
            # Generate the image
            start_time = time.time()
            output = replicate.run(self.model, input=input_params)
            end_time = time.time()
            
            logger.info(f"Image generation completed in {end_time - start_time:.2f} seconds")
            
            # Handle the output - Replicate typically returns a list of URLs
            if isinstance(output, list) and len(output) > 0:
                image_url = output[0]
            elif isinstance(output, str):
                image_url = output
            else:
                raise ValueError(f"Unexpected output format from Replicate: {type(output)}")
            
            logger.info(f"Generated image URL: {image_url}")
            
            # Download the image and convert to base64
            try:
                logger.info("Downloading generated image...")
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Convert to base64
                image_data = response.content
                img_str = base64.b64encode(image_data).decode()
                
                logger.info(f"Successfully generated and downloaded image for prompt: {prompt}")
                
                return {
                    "image_base64": img_str,
                    "prompt": prompt,
                    "fallback": False,
                    "image_url": image_url
                }
                
            except Exception as download_error:
                logger.error(f"Error downloading image from URL {image_url}: {str(download_error)}")
                # Return fallback image
                image = create_text_image(f"Generated but failed to download: {prompt}", width, height)
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return {
                    "image_base64": img_str,
                    "prompt": prompt,
                    "fallback": True,
                    "image_url": image_url
                }
                
        except Exception as e:
            logger.error(f"Error generating image with Replicate: {str(e)}")
            
            # Check if it's a billing error
            error_str = str(e).lower()
            if "payment required" in error_str or "billing" in error_str or "402" in error_str:
                error_message = "💳 Billing Setup Required!\n\nPlease add a payment method at:\nhttps://replicate.com/account/billing\n\nImages cost ~$0.002 each"
            else:
                error_message = f"Replicate API error: {str(e)[:100]}"
            
            # Return fallback image with error message
            try:
                image = create_text_image(f"⚠️ Image Generation Failed\n\n{error_message}\n\nPrompt: {prompt}", width, height)
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return {
                    "image_base64": img_str,
                    "prompt": prompt,
                    "fallback": True,
                    "image_url": None
                }
            except Exception as img_error:
                logger.error(f"Error creating fallback image: {str(img_error)}")
                raise RuntimeError(f"Complete failure in image generation: {str(e)}")
    
    def refine_prompt(self, prompt):
        """
        Simple pass-through function that cleans the prompt.
        Prompt enhancement is now handled by Together AI in main.py.
        
        Args:
            prompt: The original prompt
            
        Returns:
            Cleaned prompt
        """
        # Just clean the prompt and remove common prefixes
        prompt = prompt.strip()
        
        # Remove common prefixes that users might add
        prefixes_to_remove = ["generate image of", "create image of", "make image of", "draw", "paint"]
        for prefix in prefixes_to_remove:
            if prompt.lower().startswith(prefix):
                prompt = prompt[len(prefix):].strip()
                
        logger.info(f"Using cleaned prompt: {prompt}")
        return prompt
        
    def count_tokens(self, text):
        """
        Estimate token count of a text string.
        This is a simple estimation - each word and punctuation is roughly one token.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return len(tokens)
        
    def truncate_prompt(self, prompt, max_tokens=200):
        """
        Intelligently truncate a prompt to fit within token limits.
        
        Args:
            prompt: The prompt to truncate
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated prompt
        """
        # If already under limit, return as is
        if self.count_tokens(prompt) <= max_tokens:
            return prompt
            
        logger.warning(f"Prompt exceeds token limit ({self.count_tokens(prompt)} > {max_tokens}). Truncating...")
        
        # Split into words and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', prompt)
        
        # Keep most important tokens (earlier ones tend to be more important)
        truncated_tokens = tokens[:max_tokens-3]
        truncated_prompt = ' '.join(truncated_tokens).replace(' ,', ',').replace(' .', '.').replace(' ;', ';')
        
        logger.info(f"Truncated prompt from {len(tokens)} to {len(truncated_tokens)} tokens")
        logger.info(f"Truncated prompt: {truncated_prompt}")
        
        return truncated_prompt
        
    def smart_prompt_optimization(self, prompt, max_tokens=200):
        """
        Advanced prompt optimization that prioritizes important elements.
        
        Args:
            prompt: The original prompt
            max_tokens: Maximum tokens allowed
            
        Returns:
            Optimized prompt within token limits
        """
        if self.count_tokens(prompt) <= max_tokens:
            return prompt
            
        logger.info("Applying smart prompt optimization...")
        
        # Common quality/style modifiers to preserve
        important_modifiers = [
            'high quality', 'detailed', 'realistic', 'photorealistic', 'masterpiece',
            'best quality', 'ultra detailed', '4k', '8k', 'hdr', 'cinematic',
            'professional', 'sharp focus', 'beautiful', 'stunning', 'vibrant'
        ]
        
        # Split prompt into parts
        parts = prompt.split(',')
        parts = [part.strip() for part in parts if part.strip()]
        
        # Start with the first part (usually the main subject)
        optimized_parts = []
        current_tokens = 0
        
        # Add parts in order of importance
        for i, part in enumerate(parts):
            part_tokens = self.count_tokens(part)
            
            # Always include the first part (main subject)
            if i == 0:
                optimized_parts.append(part)
                current_tokens += part_tokens
            # Check if adding this part would exceed limit
            elif current_tokens + part_tokens + 2 <= max_tokens:  # +2 for comma and space
                optimized_parts.append(part)
                current_tokens += part_tokens + 2
            # If we can't fit the whole part, see if it contains important modifiers
            else:
                # Check if this part contains important modifiers
                part_lower = part.lower()
                for modifier in important_modifiers:
                    if modifier in part_lower and current_tokens + self.count_tokens(modifier) + 2 <= max_tokens:
                        optimized_parts.append(modifier)
                        current_tokens += self.count_tokens(modifier) + 2
                        break
                
                # Stop adding parts if we're getting close to limit
                if current_tokens >= max_tokens - 5:
                    break
        
        optimized_prompt = ', '.join(optimized_parts)
        
        # Final check and truncation if needed
        if self.count_tokens(optimized_prompt) > max_tokens:
            optimized_prompt = self.truncate_prompt(optimized_prompt, max_tokens)
        
        logger.info(f"Optimized prompt: {optimized_prompt}")
        return optimized_prompt

    @staticmethod
    def test_connection(api_token=None):
        """
        Test the Replicate API connection.
        
        Args:
            api_token: Optional API token to test with
            
        Returns:
            Dict with connection test results
        """
        try:
            if api_token:
                os.environ["REPLICATE_API_TOKEN"] = api_token
            
            # Try to list models to test connection
            models = replicate.models.list()
            model_count = len(list(models))
            
            return {
                "success": True,
                "message": f"Successfully connected to Replicate API. Found {model_count} models.",
                "api_token_set": bool(os.environ.get("REPLICATE_API_TOKEN"))
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to connect to Replicate API: {str(e)}",
                "api_token_set": bool(os.environ.get("REPLICATE_API_TOKEN"))
            }
