from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
# from pydantic_ai.models.gemini import GeminiModel  # Uncomment if you want Gemini support
from models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt, GeneratedImage
import base64
from typing import Optional


"""
NOTE TO MYSELF:

-> async:
In computer programming, asynchronous (async) operation means that a process operates independent of other processes ("non-blocking"). Synchronous (sync) operation means that the process runs only as a result of some other completed or handed-off processes ("blocking").

-> How to read base64.b64encode(image_file.read()).decode("utf-8")? (same for "image_bytes" as param):
Plain English: "Take these image bytes or open this image file and read its content as bytes, then convert them to base64 format, then turn the base64 bytes into a readable string. But we need a string for the API, so decode the bytes to UTF-8 (Unicode Transformation Format 8-bit) string."
"""


class Agents:
    """
    Encapsulates all AI agent logic for product image analysis, moodboard analysis, user vision parsing, prompt building, and image generation. Supports multiple LLM providers and dynamic model selection. All methods are async and ready for FastAPI integration.
    """
    def __init__(
        self,
        text_openai_api_key: Optional[str] = None, # For GPT calls (MS_OPENAI_API_KEY)
        img_openai_api_key: Optional[str] = None, # For GPT image calls (MY_OPENAI_API_KEY)
        text_model_name: str = "gpt-4o-mini", # For text
        img_model_name: str = "gpt-4.1", # For image generation
        provider: str = "openai"
    ):
        """
    Initialize the Agents class with API keys, model names, and provider.
    Sets up all agent instances for product image analysis, moodboard analysis, user vision parsing, prompt building, and image generation.

    Args:
        text_openai_api_key (Optional[str]): API key for OpenAI text models (GPT calls).
        img_openai_api_key (Optional[str]): API key for OpenAI image generation models (gpt-image-1 calls).
        text_model_name (str): The model name for text operations (default: 'openai:gpt-4o-mini').
        img_model_name (str): The model name for image generation operations (default: 'openai:gpt-4.1').
        provider (str): The provider name (default: 'openai').
    """
        self.text_openai_api_key = text_openai_api_key
        self.img_openai_api_key = img_openai_api_key
        self.text_model_name = text_model_name
        self.img_model_name = img_model_name
        self.provider = provider
        self._init_agents()


    def _get_text_model(self):
        """
        Get the OpenAI model for text-based operations (GPT calls).
        
        Returns:
            OpenAIModel: Configured model for text analysis and prompt generation.
            
        Raises:
            ValueError: If provider is not supported.
        """
        if self.provider == "openai":
            client = AsyncOpenAI(api_key=self.text_openai_api_key)
            return OpenAIModel(self.text_model_name, provider=OpenAIProvider(openai_client=client))
        # elif self.provider == "gemini":
        #     return GeminiModel(self.model_name, api_key=self.gemini_api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


    def _get_image_model(self):
        """
        Get the OpenAI model for image generation operations (gpt-image-1 calls).
        
        Returns:
            OpenAIModel: Configured model for image generation.
            
        Raises:
            ValueError: If provider is not supported.
        """
        if self.provider == "openai":
            client = AsyncOpenAI(api_key=self.img_openai_api_key)
            return OpenAIModel(self.img_model_name, provider=OpenAIProvider(openai_client=client))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


    def _init_agents(self):
        text_model = self._get_text_model()
        image_model = self._get_image_model()

        self.product_image_agent = Agent(
            text_model,
            output_type=ImageAnalysis,
            system_prompt=(
                "You are an expert visual analyst for advertising and social media. "
                "Analyze the product image and return a structured JSON with the following fields: "
                "product_type, product_category, style_descriptors, material_details, distinctive_features, "
                "primary_colors, accent_colors, brand_elements, advertising_keywords, overall_aesthetic.\n"
                "For each field, follow these examples:\n"
                "- product_type: 'sneakers', 'dress shirt', 'backpack'\n"
                "- product_category: 'footwear', 'apparel', 'accessories', 'electronics'\n"
                "- style_descriptors: ['minimalist', 'low-top'], ['vintage', 'elegant']\n"
                "- material_details: ['leather', 'mesh'], ['cotton', 'denim']\n"
                "- distinctive_features: ['white sole', 'perforated toe']\n"
                "- primary_colors: ['black', 'white'], ['navy blue', 'gray']\n"
                "- accent_colors: ['red accents', 'silver details']\n"
                "- brand_elements: ['Puma logo', 'embossed text'], ['Nike logo', 'swoosh']\n"
                "- advertising_keywords: ['urban', 'athletic', 'versatile']\n"
                "- overall_aesthetic: 'luxury minimalist', 'urban casual'\n"
                "Return only the structured JSON, no explanation."
            )
        )
        self.moodboard_agent = Agent(
            text_model,
            output_type=MoodboardAnalysis,
            system_prompt=(
                "You are an expert moodboard analyst for advertising and social media. "
                "Analyze the moodboard image and return a structured JSON with the following fields: "
                "scene_description, visual_style, mood_atmosphere, color_theme, composition_patterns, suggested_keywords.\n"
                "For each field, follow these examples:\n"
                "- scene_description: 'A group of friends laughing in a cozy living room'\n"
                "- visual_style: 'warm, inviting, casual'\n"
                "- mood_atmosphere: 'relaxed, joyful'\n"
                "- color_theme: ['beige', 'soft blue', 'warm yellow']\n"
                "- composition_patterns: 'central focus, natural light'\n"
                "- suggested_keywords: ['weekend', 'friendship', 'comfort', 'home', 'laughter']\n"
                "Return only the structured JSON, no explanation."
            )
        )
        self.user_vision_agent = Agent(
            text_model,
            output_type=UserVision,
            system_prompt=(
                "You are an expert at extracting structured information from user descriptions for advertising content creation. "
                "Extract: subjects, action, setting, lighting, mood_descriptors, additional_details.\n"
                "For each field, follow these examples:\n"
                "- subjects: 'young teenage girl', 'professional woman', 'group of friends'\n"
                "- action: 'skating', 'drinking coffee', 'laughing'\n"
                "- setting: 'Venice Beach skatepark', 'urban coffee shop', 'cozy living room'\n"
                "- lighting: 'blue hour', 'morning golden hour', 'warm evening light'\n"
                "- mood_descriptors: ['confident', 'relaxed'], ['joyful', 'casual']\n"
                "- additional_details: ['graffiti-covered', 'minimalist white sneakers'], ['city grit', 'weekend vibe']\n"
                "Return only the structured JSON, no explanation."
            )
        )
        self.prompt_agent = Agent(
            text_model,
            output_type=Prompt,
            system_prompt=(
                "You are an expert at creating optimized prompts for OpenAI's image generation tool for advertising imagery. "
                "Given product analysis, moodboard inspiration, user vision, and a focus instruction, "
                "create a single, cohesive prompt (30-75 words) that ensures the product is visible and identifiable.\n"
                "Requirements:\n"
                "- Use specific details from the analysis.\n"
                "- The prompt should be 30-75 words.\n"
                "- Include photography/cinematography terms when appropriate.\n"
                "- The product MUST be visible and identifiable in the final image.\n"
                "- Return only the prompt text, no explanation."
            )
        )
        self.image_gen_agent = Agent(
            image_model,
            output_type=GeneratedImage,
            system_prompt=(
                "You are an expert at generating advertising images from prompts and reference images using OpenAI's image generation API. "
                "Given a prompt and one or more input images, generate a realistic advertising image.\n"
                "Return only the generated image data, no explanation."
            )
        )


    def set_model(self, model_name: str, provider: str = "openai"):
        """
        Change the model and provider for all agents. Re-initializes all agent instances.

        Args:
            model_name (str): The new model name to use.
            provider (str): The provider name (e.g., 'openai', 'gemini').
        """
        self.model_name = model_name
        self.provider = provider
        self._init_agents()


    async def analyze_product_image(self, base64_image: str) -> ImageAnalysis:
        """
        Analyze a product image using the product_image_agent and return structured analysis.

        Args:
            image_path (str): Path to the product image file.
        Returns:
            ImageAnalysis: Structured analysis of the product image.
        """
        prompt = [
            {
                "type": "input_text",
                "text": (
                    "Analyze this product image for advertising purposes and provide:\n"
                    "1. Product type (e.g., 'sneakers', 'dress shirt', 'backpack')\n"
                    "2. Product category (e.g., 'footwear', 'apparel', 'accessories', 'electronics')\n"
                    "3. Style descriptors as list (e.g., ['minimalist', 'low-top'], ['vintage', 'elegant'])\n"
                    "4. Material details as list (e.g., ['leather', 'mesh'], ['cotton', 'denim'])\n"
                    "5. Distinctive features as list (e.g., ['white sole', 'perforated toe'])\n"
                    "6. Primary colors as list (e.g., ['black', 'white'], ['navy blue', 'gray'])\n"
                    "7. Accent colors as list (e.g., ['red accents', 'silver details'])\n"
                    "8. Brand elements as list (e.g., ['Puma logo', 'embossed text'], ['Nike logo', 'swoosh'])\n"
                    "9. Advertising keywords as list (e.g., ['urban', 'athletic', 'versatile'])\n"
                    "10. Overall aesthetic (optional) (e.g., 'luxury minimalist', 'urban casual')\n"
                    "Return only the structured JSON, no explanation."
                )
            },
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
            }
        ]
        result = await self.product_image_agent.run(prompt)
        return result.output


    async def analyze_moodboard(self, base64_images: list[str]) -> list[MoodboardAnalysis]:
        """
        Analyze a list of moodboard images using the moodboard_agent and return structured analyses.

        Args:
            image_paths (list[str]): List of paths to moodboard image files.
        Returns:
            list[MoodboardAnalysis]: List of structured analyses for each moodboard image.
        """
        results = []
        for base64_image in base64_images:
            prompt = [
                {
                    "type": "input_text",
                    "text": (
                        "Analyze this moodboard image and provide:\n"
                        "1. Brief scene description (e.g., 'A group of friends laughing in a cozy living room')\n"
                        "2. Visual style (e.g., 'warm, inviting, casual')\n"
                        "3. Mood/atmosphere (e.g., 'relaxed, joyful')\n"
                        "4. Color theme (e.g., ['beige', 'soft blue', 'warm yellow'])\n"
                        "5. Composition patterns (e.g., 'central focus, natural light')\n"
                        "6. 5-7 relevant keywords (e.g., ['weekend', 'friendship', 'comfort', 'home', 'laughter'])\n"
                        "Return only the structured JSON, no explanation."
                    )
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                }
            ]
            result = await self.moodboard_agent.run(prompt)
            results.append(result.output)
        return results

    async def parse_user_vision(self, user_text: str) -> UserVision:
        """
        Parse a user vision description and return structured information using the user_vision_agent.

        Args:
            user_text (str): The user's vision or description as text.
        Returns:
            UserVision: Structured user vision information.
        """
        prompt = (
            f"Extract structured information from this user description: '{user_text}'\n"
            "Please identify and extract:\n"
            "1. Who: People/subjects described (e.g., 'young teenage girl', 'professional woman', 'group of friends')\n"
            "2. What: Activities, actions, or behaviors mentioned (e.g., 'skating', 'drinking coffee', 'laughing')\n"
            "3. Where: Locations, settings, or environments described (e.g., 'Venice Beach skatepark', 'urban coffee shop', 'cozy living room')\n"
            "4. When: Time of day, season, or temporal context (e.g., 'blue hour', 'morning golden hour', 'warm evening light')\n"
            "5. Mood descriptors: Any mood, style, or atmosphere words (e.g., ['confident', 'relaxed'], ['joyful', 'casual'])\n"
            "6. Additional details: Any other specific requests or requirements (e.g., ['graffiti-covered', 'minimalist white sneakers'], ['city grit', 'weekend vibe'])\n"
            "If any category is not mentioned or unclear, leave it empty or mark as 'not specified'.\n"
            "Return only the structured JSON, no explanation."
        )
        result = await self.user_vision_agent.run(prompt)
        return result.output


    async def build_advertising_prompt(
        self,
        image_analysis: ImageAnalysis,
        moodboard_analyses: list[MoodboardAnalysis],
        user_vision: UserVision,
        focus_slider: int
    ) -> Prompt:
        """
        Build a single, cohesive prompt for OpenAI's image generation tool using the provided analysis data and focus instruction.

        Args:
            image_analysis (ImageAnalysis): Structured product image analysis.

            moodboard_analyses (list[MoodboardAnalysis]): List of structured moodboard analyses.

            user_vision (UserVision): Structured user vision information.

            focus_slider (int): Value (0-10) indicating the desired product/scene focus.
            
        Returns:
            Prompt: The generated advertising prompt.
        """
        focus_instructions = [
            "The product is the sole focus, with minimal background or scene elements.",
            "The product is the main focus, with a subtle hint of the scene for context.",
            "The product is prominent, but the scene provides gentle support.",
            "The product is clearly the hero, but the scene is present and meaningful.",
            "The product and scene are balanced, each drawing equal attention.",
            "The scene and product are equally important, blending together.",
            "The scene is slightly more prominent, but the product remains clearly visible.",
            "The scene is dominant, with the product naturally integrated and visible.",
            "The scene is the main focus, with the product subtly present.",
            "The scene is highly dominant, with the product as a supporting element.",
            "The atmosphere and setting are the sole focus, with the product barely visible but still present."
        ]
        focus_instruction = focus_instructions[min(max(focus_slider, 0), len(focus_instructions)-1)]
        
        # Combine all moodboard analyses into a single description
        moodboard_descriptions = []
        for i, moodboard in enumerate(moodboard_analyses, 1):
            moodboard_descriptions.append(
                f"MOODBOARD {i}: {moodboard.model_dump_json(indent=2)}"
            )
        combined_moodboards = "\n".join(moodboard_descriptions)
        
        prompt = (
            "Create a single, cohesive prompt for OpenAI's image generation tool using the following data:\n"
            f"PRODUCT ANALYSIS: {image_analysis.model_dump_json(indent=2)}\n"
            f"MOODBOARD INSPIRATIONS:\n{combined_moodboards}\n"
            f"USER VISION: {user_vision.model_dump_json(indent=2)}\n"
            f"FOCUS INSTRUCTION: {focus_instruction}\n"
            "Requirements:\n"
            "- The prompt should be 30-75 words.\n"
            "- Use specific details from the analysis.\n"
            "- Ensure the product is visible and identifiable.\n"
            "- Use photography/cinematography terms when appropriate.\n"
            "- Return only the prompt text, no explanation."
        )
        result = await self.prompt_agent.run(prompt)
        return result.output


    async def generate_image(
        self,
        prompt: str,
        product_image_base64: str,
        reference_images_base64: list[str] | None = None
    ) -> GeneratedImage:
        """
        Generate an advertising image using OpenAI's image generation API, given a prompt and input images.

        Args:
            prompt (str): The advertising prompt text (from build_advertising_prompt).

            input_image_paths (list[str]): List of paths to input image files.
        Returns:
            GeneratedImage: The generated advertising image and metadata.
        """
        # Create final prompt that references the input images
        final_prompt = (
            f"{prompt}\n\n"
            "Use the provided reference images to ensure accuracy:\n"
            "- Product image: Use this for exact product details, colors, and branding\n"
            "- Character/reference images: Use these for pose, style, and scene elements\n"
            "Generate the image based on this description and the provided reference images."
        )
        
        # Create content array with final prompt and images
        content = [{"type": "input_text", "text": final_prompt}]
        
        # Add product image first
        content.append({
            "type": "input_image", 
            "image_url": f"data:image/jpeg;base64,{product_image_base64}"
        })
        
        # Add reference images if provided
        if reference_images_base64:
            for img in reference_images_base64:
                content.append({
                    "type": "input_image", 
                    "image_url": f"data:image/jpeg;base64,{img}"
                })
        
        result = await self.image_gen_agent.run(content)
        return result.output
