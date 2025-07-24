from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
# from pydantic_ai.models.gemini import GeminiModel  # Uncomment if you want Gemini support
from models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt, GeneratedImage
import base64
from typing import Optional, List

class Agents:
    """
    Encapsulates all AI agent logic for product image analysis, moodboard analysis, user vision parsing,
    prompt building, and image generation. Supports multiple LLM providers and dynamic model selection.
    All methods are async and ready for FastAPI integration.
    """
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        model_name: str = "openai:gpt-4o",
        provider: str = "openai"
    ):
        """
        Initialize the Agents class with API keys, model name, and provider.
        Sets up all agent instances for product image analysis, moodboard analysis, user vision parsing,
        prompt building, and image generation.

        Args:
            openai_api_key (Optional[str]): API key for OpenAI models.
            gemini_api_key (Optional[str]): API key for Gemini models (if used).
            model_name (str): The model name to use (default: 'openai:gpt-4o').
            provider (str): The provider name (default: 'openai').
        """
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self.provider = provider
        self._init_agents()

    def _get_model(self):
        if self.provider == "openai":
            return OpenAIModel(self.model_name, api_key=self.openai_api_key)
        # elif self.provider == "gemini":
        #     return GeminiModel(self.model_name, api_key=self.gemini_api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _init_agents(self):
        model = self._get_model()
        self.product_image_agent = Agent(
            model,
            result_type=ImageAnalysis,
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
            model,
            result_type=MoodboardAnalysis,
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
            model,
            result_type=UserVision,
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
            model,
            result_type=Prompt,
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
            model,
            result_type=GeneratedImage,
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

    @staticmethod
    def encode_image(image_path: str) -> str:
        """
        Encode an image file as a base64 string for API input.

        Args:
            image_path (str): Path to the image file.
        Returns:
            str: Base64-encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    async def analyze_product_image(self, image_path: str) -> ImageAnalysis:
        """
        Analyze a product image using the product_image_agent and return structured analysis.

        Args:
            image_path (str): Path to the product image file.
        Returns:
            ImageAnalysis: Structured analysis of the product image.
        """
        base64_image = self.encode_image(image_path)
        prompt = {
            "role": "user",
            "content": [
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
        }
        result = await self.product_image_agent.run([prompt])
        return result.output

    async def analyze_moodboard(self, image_path: str) -> MoodboardAnalysis:
        """
        Analyze a moodboard image using the moodboard_agent and return structured analysis.

        Args:
            image_path (str): Path to the moodboard image file.
        Returns:
            MoodboardAnalysis: Structured analysis of the moodboard image.
        """
        base64_image = self.encode_image(image_path)
        prompt = {
            "role": "user",
            "content": [
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
        }
        result = await self.moodboard_agent.run([prompt])
        return result.output

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
        moodboard_analysis: MoodboardAnalysis,
        user_vision: UserVision,
        focus_slider: int
    ) -> Prompt:
        """
        Build a single, cohesive prompt for OpenAI's image generation tool using the provided analysis data and focus instruction.

        Args:
            image_analysis (ImageAnalysis): Structured product image analysis.
            moodboard_analysis (MoodboardAnalysis): Structured moodboard analysis.
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
        prompt = (
            "Create a single, cohesive prompt for OpenAI's image generation tool using the following data:\n"
            f"PRODUCT ANALYSIS: {image_analysis.model_dump_json(indent=2)}\n"
            f"MOODBOARD INSPIRATION: {moodboard_analysis.model_dump_json(indent=2)}\n"
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
        input_image_paths: List[str]
    ) -> GeneratedImage:
        """
        Generate an advertising image using OpenAI's image generation API, given a prompt and input images.

        Args:
            prompt (str): The advertising prompt text.
            input_image_paths (List[str]): List of paths to input image files.
        Returns:
            GeneratedImage: The generated advertising image and metadata.
        """
        input_images_base64 = [self.encode_image(path) for path in input_image_paths]
        prompt_content = [
            {"type": "input_text", "text": prompt}
        ] + [
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img}"}
            for img in input_images_base64
        ]
        user_prompt = {
            "role": "user",
            "content": prompt_content
        }
        result = await self.image_gen_agent.run([user_prompt])
        return result.output
