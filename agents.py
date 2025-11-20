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
import prompts


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
            system_prompt=prompts.PRODUCT_ANALYSIS_SYSTEM_PROMPT
        )
        self.moodboard_agent = Agent(
            text_model,
            output_type=MoodboardAnalysis,
            system_prompt=prompts.MOODBOARD_ANALYSIS_SYSTEM_PROMPT
        )
        self.user_vision_agent = Agent(
            text_model,
            output_type=UserVision,
            system_prompt=prompts.USER_VISION_SYSTEM_PROMPT
        )
        self.prompt_agent = Agent(
            text_model,
            output_type=Prompt,
            system_prompt=prompts.PROMPT_ENGINEER_SYSTEM_PROMPT
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
        content = [
            prompts.PRODUCT_ANALYSIS_TASK_PROMPT,
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
            content = [
                prompts.MOODBOARD_ANALYSIS_TASK_PROMPT,
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
        prompt = prompts.USER_VISION_TASK_TEMPLATE.format(user_text=user_text)
        
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
        # 1. Use static list from prompts.py
        # Ensure slider is within bounds (0-10)
        idx = min(max(focus_slider, 0), len(prompts.FOCUS_INSTRUCTIONS) - 1)
        focus_instruction = prompts.FOCUS_INSTRUCTIONS[idx]
        
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
        
        # 2. Use the template from prompts.py
        # We handle the JSON formatting here (data layer) before passing to the template (presentation layer)
        prompt = prompts.AD_PROMPT_TEMPLATE.format(
            examples_section=examples_section,
            product_analysis=json.dumps(image_analysis, indent=2),
            moodboard_inspirations=combined_moodboards,
            user_vision=json.dumps(user_vision, indent=2),
            focus_instruction=focus_instruction
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
