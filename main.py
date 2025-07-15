import os, yaml
import logging
import sys
import time
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from research_orchestrator import ResearchOrchestrator
from replicate_image_generator import ReplicateImageGenerator, create_text_image
from huggingface_image_generator import HuggingFaceImageGenerator
from fastapi.responses import JSONResponse
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("configs/open_deep_researcher_config.yaml") as f:
    config = yaml.safe_load(f)

# Set up Replicate API token from config
replicate_api_key = config.get('replicate_api_key')
if replicate_api_key and replicate_api_key != "your_replicate_api_key_here":
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    logger.info("Replicate API token loaded from config")
else:
    logger.warning("Replicate API token not found in config. Image generation will use fallback mode.")

class ResearchRequest(BaseModel):
    topic: str
    fast_mode: bool = False  # Default to comprehensive mode for detailed research

class FollowUpRequest(BaseModel):
    question: str
    original_research: Dict[str, Any]

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = None
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50  # Replicate models typically use 50 steps
    guidance_scale: float = 7.5

class PromptRefinementRequest(BaseModel):
    prompt: str

# Initialize research orchestrator
orchestrator = ResearchOrchestrator()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000", 
        "http://localhost:3001", 
        "http://127.0.0.1:3001",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "*"  # Allow all origins for testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize resources on server startup"""
    logger.info("Server starting up...")
    logger.info("Using Hugging Face API for image generation - completely free!")

@app.get("/api/health")
async def health_check():
    """Simple health check endpoint to test API connectivity and response format"""
    return JSONResponse(
        content={
            "status": "success",
            "message": "API is healthy",
            "data": {
                "server": "open-deep-research-web",
                "version": "1.0.0",
                "api_mode": "cloud",
                "image_generation": "replicate",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "features": {
                    "fast_research": True,
                    "comprehensive_research": True,
                    "image_generation": True
                }
            }
        },
        status_code=200,
        headers={
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.post("/api/research/quick")
async def quick_research(req: ResearchRequest):
    """Quick research endpoint with minimal processing for fastest results"""
    try:
        logger.info(f"Received quick research request for topic: {req.topic}")
        
        # Force fast mode for this endpoint
        research_results = await asyncio.wait_for(
            orchestrator.conduct_research(req.topic, fast_mode=True),
            timeout=60.0  # 1 minute for quick research
        )
        
        # Generate follow-up questions quickly
        follow_up_questions = [
            f"What are the latest developments in {req.topic}?",
            f"How does {req.topic} compare to alternatives?",
            f"What are the future implications of {req.topic}?",
            f"What challenges does {req.topic} face?",
            f"What are the practical applications of {req.topic}?"
        ]
        
        logger.info("Quick research completed successfully")
        
        response_data = {
            "status": "success",
            "report": research_results.get('full_report', ''),  # Frontend expects 'report' key
            "summary": research_results.get('summary', ''),
            "suggested_questions": follow_up_questions,
            "metadata": research_results.get('metadata', {})
        }
        
        return JSONResponse(
            content=response_data,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except asyncio.TimeoutError:
        logger.error("Quick research request timed out")
        error_response = {
            "status": "error",
            "message": "Quick research timed out. Please try a simpler query.",
            "data": None
        }
        return JSONResponse(
            content=error_response,
            status_code=408,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except Exception as e:
        logger.error(f"Error in quick research: {str(e)}")
        error_response = {
            "status": "error",
            "message": str(e),
            "data": None
        }
        return JSONResponse(
            content=error_response,
            status_code=500,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )

@app.options("/api/research")
async def research_options():
    """Handle CORS preflight requests for research endpoint"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

@app.post("/api/research")
async def research(req: ResearchRequest):
    """
    Endpoint to conduct comprehensive research using multiple models.
    """
    try:
        logger.info(f"Received research request for topic: {req.topic} (fast_mode: {req.fast_mode})")
        
        # Set a timeout for the research operation
        try:
            # Use asyncio.wait_for to add a timeout
            research_results = await asyncio.wait_for(
                orchestrator.conduct_research(req.topic, req.fast_mode),
                timeout=120.0  # 2 minutes maximum
            )
        except asyncio.TimeoutError:
            logger.error("Research request timed out after 2 minutes")
            error_response = {
                "status": "error",
                "message": "Research request timed out. Please try with a simpler query or enable fast mode.",
                "data": None
            }
            return JSONResponse(
                content=error_response,
                status_code=408,  # Request Timeout
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        # Generate follow-up questions
        follow_up_questions = await orchestrator.generate_follow_up_questions(research_results)
        
        logger.info("Research completed successfully")
        
        # Ensure response is properly formatted for frontend compatibility
        response_data = {
            "status": "success",
            "report": research_results.get('full_report', ''),  # Frontend expects 'report' key
            "summary": research_results.get('summary', ''),
            "suggested_questions": follow_up_questions,
            "metadata": research_results.get('metadata', {})
        }
        
        logger.info(f"Sending response with keys: {list(response_data.keys())}")
        return JSONResponse(
            content=response_data,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing research request: {str(e)}")
        error_response = {
            "status": "error",
            "message": str(e),
            "data": None
        }
        return JSONResponse(
            content=error_response,
            status_code=500,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )

@app.post("/api/follow-up")
async def follow_up(req: FollowUpRequest):
    """
    Endpoint to handle follow-up questions about the research.
    """
    try:
        logger.info(f"Received follow-up question: {req.question}")
        
        if not req.original_research:
            logger.warning("No original research provided for context")
            raise HTTPException(400, detail="No original research provided for context")
        
        # Process the follow-up question
        answer = await orchestrator.process_follow_up(
            req.original_research,
            req.question
        )
        
        logger.info("Follow-up question answered successfully")
        return {
            "status": "success",
            "question": req.question,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Error processing follow-up request: {str(e)}")
        raise HTTPException(500, detail=str(e))

@app.post("/api/refine-prompt")
async def refine_prompt(req: PromptRefinementRequest):
    try:
        logger.info(f"Received prompt refinement request: {req.prompt}")
        
        if not req.prompt.strip():
            raise HTTPException(400, detail="Image prompt cannot be empty")
        
        # Use the research orchestrator for prompt refinement
        refinement_prompt = f"""You are an expert prompt engineer specializing in image generation. Your task is to enhance and refine the following user prompt to produce better, more detailed, and visually appealing images.

Original prompt: "{req.prompt}"

Please enhance this prompt by:
1. Adding specific artistic details (style, lighting, composition, color palette)
2. Including technical specifications for better quality (4K, high resolution, detailed)
3. Adding relevant artistic styles or references if appropriate
4. Making the description more vivid and specific
5. Ensuring the prompt is optimized for AI image generation

Return ONLY the refined prompt without any explanations or additional text. The refined prompt should be a single, well-crafted sentence or paragraph that will generate a stunning image.

Refined prompt:"""
        
        # Use the analysis model for prompt refinement
        refined_prompt = await orchestrator._execute_research_step('analysis', refinement_prompt)
        
        # Clean the response to get just the refined prompt
        refined_prompt = refined_prompt.strip()
        if refined_prompt.startswith("Refined prompt:"):
            refined_prompt = refined_prompt.replace("Refined prompt:", "").strip()
        
        logger.info(f"Refined prompt: {refined_prompt}")
        return {"original_prompt": req.prompt, "refined_prompt": refined_prompt}
        
    except Exception as e:
        logger.error(f"Error refining prompt: {str(e)}")
        raise HTTPException(500, detail=str(e))

@app.options("/api/generate-image")
async def generate_image_options():
    """Handle CORS preflight requests for image generation endpoint"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

@app.post("/api/generate-image")
async def generate_image(req: ImageRequest):
    try:
        logger.info(f"Received image generation request for prompt: {req.prompt}")
        
        if not req.prompt.strip():
            raise HTTPException(400, detail="Image prompt cannot be empty")
        
        # Initialize Hugging Face Image Generator (free alternative)
        logger.info("Initializing Hugging Face Image Generator...")
        hf_generator = HuggingFaceImageGenerator(
            api_token=config.get('huggingface_api_key'),
            model=config.get('huggingface_image_model', 'stabilityai/stable-diffusion-xl-base-1.0'),
            max_tokens=300
        )
        
        # Use Together AI for prompt enhancement with proper fallback
        try:
            logger.info("Attempting to enhance prompt with Together AI...")
            
            enhancement_prompt = f"""You are an expert prompt engineer specializing in AI image generation. Your task is to enhance and refine the following user prompt to produce better, more detailed, and visually appealing images.

Original prompt: "{req.prompt}"

Please enhance this prompt by:
1. If multiple subjects are mentioned (like "cat and dog"), ensure BOTH are clearly specified and emphasized
2. Adding specific artistic details (style, lighting, composition, color palette)
3. Including technical specifications for better quality (highly detailed, sharp focus, 4K resolution)
4. Adding relevant artistic styles or references if appropriate
5. Making the description more vivid and specific
6. For multiple subjects, use phrases like "both a [subject1] and a [subject2]" or "[subject1] alongside [subject2]"

IMPORTANT: 
- Return ONLY the enhanced prompt without any explanations, prefixes, or additional text
- Be concise as the model can only process about 75 tokens (roughly 60-70 words)
- For multiple subjects, prioritize clarity about BOTH subjects being present
- Keep your response under 300 characters for optimal results

Enhanced prompt:"""
            
            # Use Together AI for prompt enhancement
            orchestrator_result = await orchestrator._execute_research_step('analysis', enhancement_prompt)
            
            # Log the raw response for debugging
            logger.info(f"Raw Together AI response: {orchestrator_result[:100]}{'...' if len(orchestrator_result) > 100 else ''}")
            
            # Validate the Together AI response
            if (orchestrator_result and 
                len(orchestrator_result.strip()) > 10 and  # Ensure it's not too short
                not any(error_keyword in orchestrator_result.lower() for error_keyword in 
                    ['error', 'could not be completed', 'technical issues', 'unavailable', 'failed', 'unable to', 'cannot'])):
                
                # Clean the response to get just the enhanced prompt
                refined_prompt = orchestrator_result.strip()
                
                # Remove common prefixes that might be added
                prefixes_to_remove = ["enhanced prompt:", "refined prompt:", "improved prompt:", "final prompt:"]
                for prefix in prefixes_to_remove:
                    if refined_prompt.lower().startswith(prefix):
                        refined_prompt = refined_prompt[len(prefix):].strip()
                
                # Truncate if still too long
                if len(refined_prompt) > 1000:
                    logger.warning(f"Truncating overly long prompt ({len(refined_prompt)} chars)")
                    refined_prompt = refined_prompt[:997] + "..."
                    
                # Count approximate tokens (rough estimation)
                word_count = len(refined_prompt.split())
                if word_count > 60:
                    logger.warning(f"Prompt may exceed token limit ({word_count} words)")
                    # Simple approach to reduce tokens while keeping important content
                    words = refined_prompt.split()
                    refined_prompt = " ".join(words[:60])
                    logger.info(f"Truncated to approximately 60 words: {refined_prompt}")
                
                # Ensure the enhanced prompt is reasonable (not too short)
                if len(refined_prompt) >= 20:
                    logger.info(f"Successfully enhanced prompt with Together AI")
                    logger.info(f"Original: {req.prompt}")
                    logger.info(f"Enhanced: {refined_prompt}")
                else:
                    logger.warning(f"Together AI response too short ({len(refined_prompt)} chars), using fallback")
                    refined_prompt = hf_generator.refine_prompt(req.prompt)
            else:
                logger.warning(f"Together AI returned invalid response: {orchestrator_result}")
                logger.info("Using cleaned prompt without enhancement")
                refined_prompt = hf_generator.refine_prompt(req.prompt)
                
        except Exception as e:
            logger.warning(f"Error enhancing prompt with Together AI: {str(e)}")
            logger.info("Using cleaned prompt without enhancement")
            refined_prompt = hf_generator.refine_prompt(req.prompt)
        
        # Generate the image with Hugging Face API (with 2-minute timeout)
        try:
            logger.info("Starting image generation with Hugging Face API...")
            
            # Use loop.run_in_executor for Python 3.8 compatibility
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,  # Use default ThreadPoolExecutor
                    lambda: hf_generator.generate_image(
                        prompt=refined_prompt,
                        height=req.height,
                        width=req.width,
                        steps=req.num_inference_steps,
                        guidance=req.guidance_scale
                    )
                ),
                timeout=120.0  # 2 minutes timeout for Hugging Face API
            )
        except asyncio.TimeoutError:
            logger.error("Image generation timed out after 2 minutes")
            # Return a fallback response instead of failing completely
            import base64
            from io import BytesIO
            
            fallback_image = create_text_image(
                f"Image generation timed out for: {refined_prompt}\n\nThe Hugging Face API took longer than expected.\nPlease try again with a simpler prompt.",
                width=req.width,
                height=req.height
            )
            buffered = BytesIO()
            fallback_image.save(buffered, format="PNG")
            fallback_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return JSONResponse(
                content={
                    "status": "timeout",
                    "image_base64": fallback_base64,
                    "original_prompt": req.prompt,
                    "refined_prompt": refined_prompt,
                    "fallback": True,
                    "message": "Image generation took too long. Please try again with a simpler prompt."
                },
                status_code=200,  # Return 200 with fallback image instead of error
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        # Log appropriate message based on success or fallback
        if result.get("fallback", False):
            logger.warning("Image generation fell back to text image due to API error")
        else:
            logger.info("Image generated successfully with Hugging Face API")
        
        response_data = {
            "status": "success",
            "image_base64": result["image_base64"],
            "original_prompt": req.prompt,
            "refined_prompt": refined_prompt,
            "fallback": result.get("fallback", False),
            "image_url": result.get("image_url")  # Include the Replicate image URL
        }
        
        return JSONResponse(
            content=response_data,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        # Check for API key errors specifically
        if "authentication" in str(e).lower() or "api_token" in str(e).lower() or "unauthorized" in str(e).lower():
            error_message = (
                f"Replicate API authentication error: {str(e)}. "
                "Please check that your Replicate API key is correctly set in the configuration file."
            )
            raise HTTPException(401, detail=error_message)
        else:
            raise HTTPException(500, detail=f"Image generation failed: {str(e)}")

@app.get("/api/test-replicate")
async def test_replicate():
    """
    Endpoint to test that Replicate API is working
    Returns information about Replicate API status and a simple test image
    """
    try:
        logger.info("Testing Hugging Face API connection")
        
        # Test connection
        connection_test = HuggingFaceImageGenerator.test_connection(config.get('huggingface_api_key'))
        
        if not connection_test["success"]:
            return {
                "status": "error",
                "error": connection_test["message"],
                "api_key_configured": connection_test["api_token_set"]
            }
        
        # Initialize Hugging Face generator
        hf_generator = HuggingFaceImageGenerator(
            api_token=config.get('huggingface_api_key'),
            model=config.get('huggingface_image_model', 'stabilityai/stable-diffusion-xl-base-1.0'),
            max_tokens=300
        )
        
        # Generate a simple test image
        test_prompt = "A test image of a mountain landscape, high quality"
        result = hf_generator.generate_image(
            prompt=test_prompt,
            height=512,
            width=512,
            steps=25  # Use fewer steps for quick testing
        )
        
        return {
            "status": "success",
            "connection_info": connection_test,
            "image_base64": result["image_base64"],
            "prompt": test_prompt,
            "image_url": result.get("image_url"),
            "fallback_used": result.get("fallback", False)
        }
    except Exception as e:
        logger.error(f"Replicate API test failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "api_key_configured": bool(config.get('replicate_api_key') and config.get('replicate_api_key') != "your_replicate_api_key_here")
        }

@app.get("/api/huggingface-status")
async def huggingface_status():
    """Simple endpoint to check Hugging Face API status"""
    connection_test = HuggingFaceImageGenerator.test_connection(config.get('huggingface_api_key'))
    
    return {
        "api_configured": bool(config.get('replicate_api_key') and config.get('replicate_api_key') != "your_replicate_api_key_here"),
        "connection_test": connection_test,
        "environment_variables": {
            key: "***" if "token" in key.lower() or "key" in key.lower() else value 
            for key, value in os.environ.items() 
            if "REPLICATE" in key
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Standard configuration for API-based image generation
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False
    )
