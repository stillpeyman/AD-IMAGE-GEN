# stdlib imports
import json

# third-party imports
from openai import AsyncOpenAI
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# local imports
from models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt


"""
NOTE TO MYSELF:

-> async:
In computer programming, asynchronous (async) operation means that a process operates independent of other processes ("non-blocking"). Synchronous (sync) operation means that the process runs only as a result of some other completed or handed-off processes ("blocking").

-> HOW TO READ base64.b64encode(image_file.read()).decode("utf-8")? (same for "image_bytes" as param):
Plain English: "Take these image bytes or open this image file and read its content as bytes, then convert them to base64 format, then turn the base64 bytes into a readable string. But we need a string for the API, so decode the bytes to UTF-8 (Unicode Transformation Format 8-bit) string."

-> UNDERSTANDING THE TWO TYPES OF PROMPTS:
    - System Prompts (in __init__): Set the agent's role, expertise, and output format. Enhanced with CoT reasoning steps.
    - Method Prompts (in each method): Task-specific instructions sent with actual data. Enhanced with Few-Shot examples.
"""


class Agents:
    """
    Handles AI text analysis tasks for the ad generation workflow:
    - Product image analysis (vision + text)
    - Moodboard analysis (vision + text)
    - User vision parsing (text)
    - Advertising prompt building (text)
    
    Supports both OpenAI and Google providers:
    - OpenAI: gpt-4.1 model
    - Google: gemini-2.5-flash model
    
    Note: Image generation is handled separately in api/image_generator.py
    """
    def __init__(
        self,
        model_provider: str,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
    ):
        """
        Initialize agents based on the selected provider.

        Args:
            model_provider: Provider selection ("openai" or "gemini").
            openai_api_key: API key for OpenAI operations (optional; required when provider="openai").
            gemini_api_key: API key for Google operations (optional; required when provider="gemini").

        Raises:
            ValueError: If a required API key is missing for the selected provider.
        """
        # Validate injected API keys based on provider at init (fail-fast approach)
        if model_provider == "openai" and not openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider.")
        if model_provider == "gemini" and not gemini_api_key:
            raise ValueError("Gemini API key is required when using Google provider.")
        
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.model_provider = model_provider
        
        # Initialize the appropriate model based on provider
        if model_provider == "openai":
            # AsyncOpenAI client needed because pydantic-ai's OpenAIModel requires it
            client = AsyncOpenAI(api_key=self.openai_api_key)
            text_model = OpenAIModel(
                "gpt-4.1", 
                provider=OpenAIProvider(openai_client=client)
            )
        elif model_provider == "gemini":
            provider = GoogleProvider(api_key=self.gemini_api_key)
            text_model = GoogleModel("gemini-2.5-flash", provider=provider)
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Initialize all text-based agents
        self._initialize_agents(text_model)


    def _initialize_agents(self, text_model):
        """
        Initialize all text-based agents with the provided model.

        Args:
            text_model: Configured text model to use.
        """

        self.product_image_agent = Agent(
            text_model,
            output_type=ImageAnalysis,
            system_prompt=(
                "You are a professional visual analyst specializing in advertising and social media. "
                "Analyze product images systematically using this methodology:\n\n"
                
                "STEP 1: Product Classification\n"
                "- Identify the specific product type and classify into one of these categories: footwear, apparel, accessories, electronics, furniture, home_goods, beauty, sports\n\n"

                "STEP 2: Physical Analysis\n" 
                "- Document style elements (silhouette, design philosophy, aesthetic approach)\n"
                "- Identify materials and textures visible in the image\n"
                "- Note distinctive features that differentiate this from similar products\n\n"

                "STEP 3: Color Analysis\n"
                "- Identify dominant colors that define the product\n"
                "- Note accent colors used for details, trim, or highlights\n\n"

                "STEP 4: Brand Elements\n"
                "- Locate all visible logos, brand marks, text, and symbols\n\n"

                "STEP 5: Advertising Positioning\n"
                "- Generate keywords based on visual appeal and target market signals\n"
                "- Define the overall aesthetic that would resonate with consumers\n\n"
                
                "OUTPUT FORMAT:\n"
                "Return structured JSON with: product_type, product_category, style_descriptors, "
                "material_details, distinctive_features, primary_colors, accent_colors, "
                "brand_elements, advertising_keywords, overall_aesthetic.\n\n"
                
                "Field examples:\n"
                "- product_type: 'running sneakers', 'dress shirt', 'gaming backpack'\n"
                "- product_category: 'footwear', 'apparel', 'accessories', 'electronics', 'furniture', 'home_goods', 'beauty', 'sports'\n"
                "- style_descriptors: ['minimalist', 'low-top'], ['vintage', 'elegant']\n"
                "- material_details: ['leather', 'mesh'], ['cotton', 'denim']\n"
                "- distinctive_features: ['white sole', 'perforated toe']\n"
                "- primary_colors: ['black', 'white'], ['navy blue', 'gray']\n"
                "- accent_colors: ['red accents', 'silver details']\n"
                "- brand_elements: ['Puma logo', 'embossed text'], ['Nike logo', 'swoosh']\n"
                "- advertising_keywords: ['urban', 'athletic', 'versatile']\n"
                "- overall_aesthetic: 'luxury minimalist', 'urban casual'\n\n"
                
                "Return only the structured JSON, no reasoning or explanation."
            )
        )
        self.moodboard_agent = Agent(
            text_model,
            output_type=MoodboardAnalysis,
            system_prompt=(
                "You are a professional moodboard analyst specializing in advertising and social media campaigns. "
                "Analyze moodboard images systematically to extract inspiration for advertising content:\n\n"

                "STEP 1: Scene Analysis\n"
                "- Describe what is happening in the image and who is present\n"
                "- Identify the primary setting, environment, and context\n\n"

                "STEP 2: Style and Mood Assessment\n"
                "- Define the visual style (aesthetic approach, design elements)\n"
                "- Capture the emotional atmosphere and overall mood\n\n"

                "STEP 3: Technical Analysis\n"
                "- Extract the color palette and dominant color themes\n"
                "- Identify composition techniques and lighting patterns\n\n"

                "STEP 4: Advertising Inspiration\n"
                "- Generate keywords that capture what brand messages this mood supports\n"
                "- Consider the campaign potential of this visual inspiration\n\n"

                "OUTPUT FORMAT:\n"
                "Return structured JSON with: scene_description, visual_style, mood_atmosphere, color_theme, composition_patterns, suggested_keywords.\n\n"

                "Field examples:\n"
                "- scene_description: 'Three young friends skateboarding at sunset in an urban plaza'\n"
                "- visual_style: 'gritty urban, authentic street culture'\n"
                "- mood_atmosphere: 'energetic, rebellious, freedom'\n"
                "- color_theme: ['warm orange', 'deep purple', 'concrete gray']\n"
                "- composition_patterns: 'dynamic angles, golden hour lighting, leading lines'\n"
                "- suggested_keywords: ['youth', 'rebellion', 'urban', 'authentic', 'energy', 'friendship']\n\n"

                "Return only the structured JSON, no reasoning or explanation."
            )
        )
        self.user_vision_agent = Agent(
            text_model,
            output_type=UserVision,
            system_prompt=(
                "You are a professional creative director specializing in translating user visions into structured advertising briefs. "
                "Parse user vision text systematically to extract actionable scene elements:\n\n"
                
                "STEP 1: Focus and Action Identification\n"
                "- Identify the primary focus of the scene (human subjects or product presentation style)\n"
                "- Determine what should be happening (activities, behaviors, product positioning)\n\n"
                
                "STEP 2: Scene Context Analysis\n"
                "- Define the setting and environment where the scene takes place\n"
                "- Specify lighting conditions and visual atmosphere\n\n"
                
                "STEP 3: Emotional and Brand Positioning\n"
                "- Extract mood descriptors and emotional tone\n"
                "- Identify additional creative requirements and brand positioning elements\n\n"
                
                "Handle any input style: specific directions, abstract concepts, or mixed descriptions. "
                "When information is not explicitly stated, make intelligent inferences based on context.\n\n"
                
                "OUTPUT FORMAT:\n"
                "Return structured JSON with: focus_subject, action, setting, lighting, mood_descriptors, additional_details.\n\n"
                
                "Field examples:\n"
                "- focus_subject: 'professional woman in her 30s', 'minimalist product showcase', 'confident teenager'\n"
                "- action: 'walking confidently', 'elegant product display', 'enjoying casual conversation'\n"
                "- setting: 'modern office environment', 'clean studio backdrop', 'cozy home workspace'\n"
                "- lighting: 'soft natural light', 'warm golden hour', 'clean studio lighting'\n"
                "- mood_descriptors: ['confident', 'sophisticated'], ['minimalist', 'premium'], ['relaxed', 'aspirational']\n"
                "- additional_details: ['luxury positioning', 'corporate appeal'], ['street culture', 'urban authenticity']\n\n"
                
                "Return only the structured JSON, no reasoning or explanation."
            )
        )
        self.prompt_agent = Agent(
            text_model,
            output_type=Prompt,
            system_prompt=(
                "You are a professional advertising prompt engineer specializing in image generation for marketing campaigns. "
                "Create advertising prompts that work with product images to generate compelling marketing visuals.\n\n"
                
                "STEP 1: Product Analysis Integration\n"
                "- Review all product analysis data for visual composition elements\n"
                "- Identify key advertising positioning and visual characteristics\n\n"
                
                "STEP 2: Scene Requirements Processing\n"
                "- Extract scene composition requirements and brand positioning from user vision\n"
                "- Incorporate moodboard inspiration when provided\n\n"
                
                "STEP 3: Focus Balance Application\n"
                "- Apply focus instruction to determine product-scene emphasis\n"
                "- Adjust composition priorities based on focus level\n\n"
                
                "STEP 4: Advertising Prompt Creation\n"
                "- Synthesize all elements into cohesive 30-75 word prompt\n"
                "- Include photography/cinematography terminology for professional results\n"
                "- Ensure product visibility and advertising effectiveness\n\n"
                
                "Return only the final prompt text, no explanation."
            )
        )
        # Note: image generation is handled via OpenAI Responses API directly (see api/image_generator.py)


    async def analyze_product_image(self, image_bytes: bytes) -> ImageAnalysis:
        """
        Analyze a product image and return structured analysis.

        Args:
            image_bytes: Raw product image data.

        Returns:
            Structured ImageAnalysis.
        """
        prompt_text = """
            EXAMPLES OF EFFECTIVE PRODUCT ANALYSIS:

            Example 1:
            {
                "product_type": "running sneakers",
                "product_category": "footwear",
                "style_descriptors": ["athletic", "low-top"],
                "material_details": ["mesh", "rubber"],
                "distinctive_features": ["air cushioning", "reflective strips"],
                "primary_colors": ["black", "white"],
                "accent_colors": ["neon green"],
                "brand_elements": ["Nike swoosh", "Air Max logo"],
                "advertising_keywords": ["performance", "urban", "versatile"],
                "overall_aesthetic": "sporty minimalist"
            }

            Example 2:
            {
                "product_type": "leather jacket",
                "product_category": "apparel",
                "style_descriptors": ["vintage", "biker"],
                "material_details": ["genuine leather", "metal hardware"],
                "distinctive_features": ["asymmetrical zipper", "studded details"],
                "primary_colors": ["black"],
                "accent_colors": ["silver hardware"],
                "brand_elements": ["brand patch", "metal studs"],
                "advertising_keywords": ["rebellious", "classic", "edgy"],
                "overall_aesthetic": "rock vintage"
            }

            Analyze this product image for advertising purposes and provide:
            1. Product type (e.g., 'sneakers', 'dress shirt', 'backpack')
            2. Product category (e.g., 'footwear', 'apparel', 'accessories', 'electronics')
            3. Style descriptors as list (e.g., ['minimalist', 'low-top'], ['vintage', 'elegant'])
            4. Material details as list (e.g., ['leather', 'mesh'], ['cotton', 'denim'])
            5. Distinctive features as list (e.g., ['white sole', 'perforated toe'])
            6. Primary colors as list (e.g., ['black', 'white'], ['navy blue', 'gray'])
            7. Accent colors as list (e.g., ['red accents', 'silver details'])
            8. Brand elements as list (e.g., ['Puma logo', 'embossed text'], ['Nike logo', 'swoosh'])
            9. Advertising keywords as list (e.g., ['urban', 'athletic', 'versatile'])
            10. Overall aesthetic (optional) (e.g., 'luxury minimalist', 'urban casual')
            Follow the format and quality shown in the examples above.
            Return only the structured JSON, no explanation.
            """
        
        content = [
            prompt_text,
            BinaryContent(data=image_bytes, media_type='image/jpeg')
        ]

        result = await self.product_image_agent.run(content)
        return result.output


    async def analyze_moodboard(self, image_bytes_list: list[bytes]) -> list[MoodboardAnalysis]:
        """
        Analyze a list of moodboard images and return structured analyses.

        Args:
            image_bytes_list: Raw moodboard image data.

        Returns:
            List of MoodboardAnalysis.
        """
        results = []

        for image_bytes in image_bytes_list:
            prompt_text = """
                EXAMPLES OF EFFECTIVE MOODBOARD ANALYSIS:

                Example 1:
                {
                    "scene_description": "Three young skaters practicing tricks at golden hour in an urban plaza",
                    "visual_style": "gritty street photography, authentic youth culture",
                    "mood_atmosphere": "energetic, rebellious, freedom",
                    "color_theme": ["warm orange", "deep shadows", "concrete gray"],
                    "composition_patterns": "dynamic angles, natural lighting, motion blur",
                    "suggested_keywords": ["youth", "rebellion", "urban", "authentic", "energy", "friendship"]
                }

                Example 2:
                {
                    "scene_description": "Professional woman enjoying morning coffee in a minimalist cafe",
                    "visual_style": "clean modern aesthetic, soft natural lighting",
                    "mood_atmosphere": "calm, sophisticated, aspirational",
                    "color_theme": ["soft whites", "warm beige", "muted gold"],
                    "composition_patterns": "clean lines, negative space, soft focus background",
                    "suggested_keywords": ["professional", "morning", "luxury", "calm", "sophisticated", "lifestyle"]
                }

                Analyze this moodboard image and provide:
                1. Brief scene description (e.g., 'A group of friends laughing in a cozy living room')
                2. Visual style (e.g., 'warm, inviting, casual')
                3. Mood/atmosphere (e.g., 'relaxed, joyful')
                4. Color theme (e.g., ['beige', 'soft blue', 'warm yellow'])
                5. Composition patterns (e.g., 'central focus, natural light')
                6. 5-7 relevant keywords (e.g., ['weekend', 'friendship', 'comfort', 'home', 'laughter'])
                Follow the format and quality shown in the examples above.
                Return only the structured JSON, no explanation.
                """

            content = [
                prompt_text,
                BinaryContent(data=image_bytes, media_type='image/jpeg')
            ]
            
            result = await self.moodboard_agent.run(content)
            results.append(result.output)

        return results


    async def parse_user_vision(self, user_text: str) -> UserVision:
        """
        Parse a user vision description and return structured information.

        Args:
            user_text: The user's vision or description as text.

        Returns:
            Structured UserVision.
        """
        # Double {{}}: Literal curly braces in f-strings
        # Single {}: Variable interpolation in f-strings
        prompt = f"""
            EXAMPLES OF EFFECTIVE USER VISION PARSING:

            Example 1:
            Input: 'Professional woman in her 30s walking confidently through modern office space during golden hour'
            Output:
            {{
                "focus_subject": "professional woman in her 30s",
                "action": "walking confidently",
                "setting": "modern office space",
                "lighting": "golden hour",
                "mood_descriptors": ["confident", "professional"],
                "additional_details": ["corporate environment", "aspirational"]
            }}

            Example 2:
            Input: 'Minimalist product showcase with clean studio lighting'
            Output:
            {{
                "focus_subject": "minimalist product showcase",
                "action": "elegant product display",
                "setting": "clean studio backdrop",
                "lighting": "clean studio lighting",
                "mood_descriptors": ["minimalist", "premium"],
                "additional_details": ["luxury positioning", "product focus"]
            }}

            Extract structured information from this user description: '{user_text}'
            Please identify and extract:
            1. Who: People/subjects described (e.g., 'young teenage girl', 'professional woman', 'group of friends')
            2. What: Activities, actions, or behaviors mentioned (e.g., 'skating', 'drinking coffee', 'laughing')
            3. Where: Locations, settings, or environments described (e.g., 'Venice Beach skatepark', 'urban coffee shop', 'cozy living room')
            4. When: Time of day, season, or temporal context (e.g., 'blue hour', 'morning golden hour', 'warm evening light')
            5. Mood descriptors: Any mood, style, or atmosphere words (e.g., ['confident', 'relaxed'], ['joyful', 'casual'])
            6. Additional details: Any other specific requests or requirements (e.g., ['graffiti-covered', 'minimalist white sneakers'], ['city grit', 'weekend vibe'])
            If any category is not mentioned or unclear, leave it empty or mark as 'not specified'.
            Follow the format and quality shown in the examples above.
            Return only the structured JSON, no explanation.
            """
        
        result = await self.user_vision_agent.run(prompt)
        return result.output


    async def build_advertising_prompt(
        self,
        image_analysis: dict,
        user_vision: dict,
        focus_slider: int,
        is_refinement: bool = False,
        moodboard_analyses: list[dict] | None = None,
        previous_prompt_text: str | None = None,
        user_feedback: str | None = None,
        prompt_examples: list[dict] | None = None
    ) -> Prompt:
        """
        Build a cohesive prompt for image generation using prior analyses and focus.

        Integrates category-matched examples for Few-Shot prompting when available.

        Args:
            image_analysis: Product image analysis as a dictionary (extracted from ImageAnalysis model).
            user_vision: User vision information as a dictionary (extracted from UserVision model).
            focus_slider: Desired balance (0–10) between product and scene.
            is_refinement: Whether this is a refinement of a previous prompt.
            moodboard_analyses: Optional list of moodboard analyses as dictionaries (extracted from MoodboardAnalysis models).
            previous_prompt_text: Previous prompt (for refinement context).
            user_feedback: Optional feedback for refinement.
            prompt_examples: Optional list of example dicts with 'prompt_text' and 'product_category' keys.

        Returns:
            Generated Prompt.
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
        if moodboard_analyses:
            for i, moodboard in enumerate(moodboard_analyses, 1):
                moodboard_descriptions.append(
                    f"MOODBOARD {i}: {json.dumps(moodboard, indent=2)}"
                )

        combined_moodboards = "\n".join(moodboard_descriptions) if moodboard_descriptions else "<no moodboards provided>"

        # Build Few-Shot examples section
        examples_section = ""
        if prompt_examples:
            examples_section = "EXAMPLES OF EFFECTIVE ADVERTISING PROMPTS:\n"
            for i, example in enumerate(prompt_examples, 1):
                # \" escapes the quote character inside the f-string
                # example is a dict: {"prompt_text": "...", "product_category": "..."}
                examples_section += f"Example {i}: \"{example['prompt_text']}\"\n"
                examples_section += "\n"
        
        prompt = (
            f"{examples_section}"
            "Create a single, cohesive prompt for OpenAI's image generation tool using the following data:\n"
            f"PRODUCT ANALYSIS: {json.dumps(image_analysis, indent=2)}\n"
            f"MOODBOARD INSPIRATIONS:\n{combined_moodboards}\n"
            f"USER VISION: {json.dumps(user_vision, indent=2)}\n"
            f"FOCUS INSTRUCTION: {focus_instruction}\n"
            "Requirements:\n"
            "- The prompt should be 30-75 words.\n"
            "- Use specific details from the analysis.\n"
            "- Ensure the product is visible and identifiable.\n"
            "- Use photography/cinematography terms when appropriate.\n"
            "- Follow the style shown in the examples above.\n"
            "- Return only the prompt text, no explanation."
        )

        if is_refinement == True:
            if user_feedback:
                refinement_instruction = (
                    f"This is a refinement iteration for the previous prompt: {previous_prompt_text}.\n"
                    f"Address this feedback: {user_feedback}."
                )
            
            else:
                refinement_instruction = (
                    f"This is a refinement iteration for the previous prompt: {previous_prompt_text}.\n"
                    "Generate a different creative interpretation."
                )
            
            result = await self.prompt_agent.run([
                prompt,
                refinement_instruction
            ])
        
        else:
            result = await self.prompt_agent.run(prompt)
        
        return result.output


