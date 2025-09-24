# stdlib imports
import base64
import logging
import os
import uuid

# third-party imports
import requests
from sqlmodel import Session, select
from sqlalchemy import func

# local imports
from agents import Agents
from api.image_generator import generate_image_data_url
from models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt, GeneratedImage, PromptExample


logger = logging.getLogger(__name__)


class AdGeneratorService:
    """
    Service layer for ad image generation workflow.
    
    Handles the complete process from image analysis to final ad generation,
    including database operations and error handling.
    """
    
    def __init__(
        self, 
        agents: Agents, 
        session: Session, 
        img_model: str,
        openai_api_key: str
    ):
        """
        Initialize the service with agents, database session, and image generation config.
        
        Args:
            agents: The AI agents for text analysis tasks (uses 'gpt-4o-mini' model with single API key)
            session: Database session for persistence
            img_model: Image generation model name (must be "gpt-image-1" - the only supported model)
            openai_api_key: OpenAI API key for image generation (MY_OPENAI_API_KEY).
                           Same key used by agents, required for all operations.
        
        Attributes:
            session_memory: Dictionary storing session data for refinement logic
        """
        # Validate injected API key at initialization
        if not openai_api_key:
            raise ValueError("OpenAI API key is required. Please set MY_OPENAI_API_KEY environment variable.")
        
        self.agents = agents
        self.session = session
        self.openai_api_key = openai_api_key
        self.img_model = img_model
        self.session_memory = {}


    @staticmethod
    def _save_image(image_bytes: bytes, dir_name: str, filename_prefix: str) -> str:
        """
        Persist an uploaded image under uploads/<dir_name>/<filename_prefix>_<uuid>.jpg
        Returns the relative path saved (string).
        """
        os.makedirs(f"uploads/{dir_name}", exist_ok=True)
        filename = f"{filename_prefix}_{uuid.uuid4().hex}.jpg"
        path = os.path.join("uploads", dir_name, filename)
        with open(path, "wb") as f:
            f.write(image_bytes)
        return path


    async def analyze_product_image(self, image_bytes: bytes, session_id: str) -> ImageAnalysis:
        """
        Analyze a product image and store results in database.
        
        Note: Uses self.agents (with pre-validated API key) for text analysis via gpt-4o-mini model.

        Args:
            image_bytes: Raw product image data
            session_id: Session identifier for linking related records
            
        Returns:
            ImageAnalysis: Database record with analysis results
            
        Raises:
            ValueError: If image analysis fails
            RuntimeError: If database operation fails
        """
        try:   
            image_path = self._save_image(image_bytes, "product", "product")
            analysis = await self.agents.analyze_product_image(image_bytes)
            
            db_analysis = ImageAnalysis(
                product_type=analysis.product_type,
                product_category=analysis.product_category,
                style_descriptors=analysis.style_descriptors,
                material_details=analysis.material_details,
                distinctive_features=analysis.distinctive_features,
                primary_colors=analysis.primary_colors,
                accent_colors=analysis.accent_colors,
                brand_elements=analysis.brand_elements,
                advertising_keywords=analysis.advertising_keywords,
                overall_aesthetic=analysis.overall_aesthetic,
                image_path=image_path,
                session_id=session_id
            )
            
            self.session.add(db_analysis)
            self.session.commit()
            self.session.refresh(db_analysis)
            
            logger.info(f"Product image analysis completed: {db_analysis.id}")
            return db_analysis
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Product image analysis failed: {str(e)}")
            raise ValueError(f"Product image analysis failed: {str(e)}")


    async def analyze_moodboard_images(
        self, 
        session_id: str, 
        image_bytes_list: list[bytes] | None = None
    ) -> list[MoodboardAnalysis]:
        """
        Analyze multiple moodboard images and store results in database.
        
        Args:
            image_bytes_list: List of raw moodboard image data
            session_id: Session identifier for linking related records
            
        Returns:
            List[MoodboardAnalysis]: Database records with analysis results
            
        Raises:
            ValueError: If moodboard analysis fails
            RuntimeError: If database operation fails
        """
        if not image_bytes_list:
            return []
        
        results = []
        
        try:
            # Save all moodboard images
            saved_paths = []
            for idx, image_bytes in enumerate(image_bytes_list):
                saved_path = self._save_image(image_bytes, "moodboards", f"moodboard_{idx+1}")
                saved_paths.append(saved_path)
            
            # Analyze all moodboard images at once
            analyses = await self.agents.analyze_moodboard([image_bytes for image_bytes in image_bytes_list])
            
            # Create database records for each analysis with corresponding image_path
            for analysis, saved_path in zip(analyses, saved_paths):
                db_analysis = MoodboardAnalysis(
                    scene_description=analysis.scene_description,
                    visual_style=analysis.visual_style,
                    mood_atmosphere=analysis.mood_atmosphere,
                    color_theme=analysis.color_theme,
                    composition_patterns=analysis.composition_patterns,
                    suggested_keywords=analysis.suggested_keywords,
                    image_path=saved_path,
                    session_id=session_id
                )
                
                self.session.add(db_analysis)
                results.append(db_analysis)
            
            self.session.commit()
            
            # Refresh each result to get IDs
            for result in results:
                self.session.refresh(result)
            
            logger.info(f"Moodboard analysis completed: {len(results)} images")
            return results
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Moodboard analysis failed: {str(e)}")
            raise ValueError(f"Moodboard analysis failed: {str(e)}")


    async def parse_user_vision(self, user_text: str, session_id: str) -> UserVision:
        """
        Parse user vision text and store results in database.
        
        Args:
            user_text: User's vision description
            session_id: Session identifier for linking related records
            
        Returns:
            UserVision: Database record with parsed vision
            
        Raises:
            ValueError: If user vision parsing fails
            RuntimeError: If database operation fails
        """
        if not user_text or not user_text.strip():
            raise ValueError("User vision text is required and cannot be empty")
        
        try:
            analysis = await self.agents.parse_user_vision(user_text)
            
            db_analysis = UserVision(
                focus_subject=analysis.focus_subject,
                action=analysis.action,
                setting=analysis.setting,
                lighting=analysis.lighting,
                mood_descriptors=analysis.mood_descriptors,
                additional_details=analysis.additional_details,
                session_id=session_id
            )
            
            self.session.add(db_analysis)
            self.session.commit()
            self.session.refresh(db_analysis)
            
            logger.info(f"User vision parsing completed: {db_analysis.id}")
            return db_analysis
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"User vision parsing failed: {str(e)}")
            raise ValueError(f"User vision parsing failed: {str(e)}")


    async def build_advertising_prompt(
        self,
        # required params first
        image_analysis_id: int,
        user_vision_id: int,
        focus_slider: int,
        session_id: str,
        # optional params last
        moodboard_analysis_ids: list[int] | None = None,
        is_refinement: bool = False,
        previous_prompt_id: int | None = None,
        user_feedback: str | None = None
    ) -> Prompt:
        """
        Build advertising prompt using analysis results, RAG examples, and store in database.

        This method retrieves relevant prompt examples from the same product category
        and passes them to the AI agent for Few-Shot learning, improving prompt quality.

        Args:
            image_analysis_id: ID of product image analysis
            user_vision_id: ID of user vision analysis  
            focus_slider: Focus level (0-10)
            session_id: Session identifier for linking related records
            moodboard_analysis_ids: List of moodboard analysis IDs (optional, defaults to None for no moodboards)
            is_refinement: Whether this is a refinement of a previous prompt (default: False)
            previous_prompt_id: ID of previous prompt for refinement context (optional)
            user_feedback: User feedback for refinement (optional)
            
        Returns:
            Prompt: Database record with generated prompt
            
        Raises:
            ValueError: If prompt building fails or required analyses not found
            RuntimeError: If database operation fails
        """
        try:
            # Get all required analyses, if not found SQLModel returns None
            product_analysis = self.session.get(ImageAnalysis, image_analysis_id)
            if not product_analysis:
                # Returning None not good for user experience, give clear error message
                # Basically business error if product analydsis is missing
                # FastAPI converts ValueError to HTTP 400 (Bad Request)
                raise ValueError(f"Image analysis with ID {image_analysis_id} not found")
            
            moodboard_analyses = []
            if moodboard_analysis_ids:
                for moodboard_id in moodboard_analysis_ids:
                    analysis = self.session.get(MoodboardAnalysis, moodboard_id)
                    if not analysis:
                        raise ValueError(f"Moodboard analysis with ID {moodboard_id} not found")
                    moodboard_analyses.append(analysis)
            
            user_vision = self.session.get(UserVision, user_vision_id)
            if not user_vision:
                raise ValueError(f"User vision with ID {user_vision_id} not found")
            
            # Calculate refinement count
            refinement_count = 0
            previous_prompt_text = None
            if is_refinement and previous_prompt_id:
                previous_prompt = self.session.get(Prompt, previous_prompt_id)
                if previous_prompt:
                    refinement_count = previous_prompt.refinement_count + 1
                    previous_prompt_text = previous_prompt.prompt_text
                    if refinement_count > 2:
                        raise ValueError("Maximum of 2 refinements allowed per prompt")
            
            # RAG: Retrieve random examples from same product category 
            # for Few-Shot prompting. If no examples returns empty list []
            stmt = select(PromptExample)\
                .where(PromptExample.product_category == product_analysis.product_category)\
                    .order_by(func.random())\
                        .limit(2)
            
            # Execute query and get up to 2 examples
            # Returns empty list [] if no examples
            prompt_examples = self.session.exec(stmt).all()

            # Use all moodboard analyses for prompt building
            prompt = await self.agents.build_advertising_prompt(
                product_analysis, 
                user_vision, 
                focus_slider,
                is_refinement,
                moodboard_analyses,
                previous_prompt_text,
                user_feedback,
                prompt_examples
            )
            
            db_prompt = Prompt(
                prompt_text=prompt.prompt_text,
                image_analysis_id=image_analysis_id,
                moodboard_analysis_ids=moodboard_analysis_ids or [],
                user_vision_id=user_vision_id,
                focus_slider=focus_slider,
                refinement_count=refinement_count,
                user_feedback=user_feedback,
                previous_prompt_id=previous_prompt_id,
                session_id=session_id
            )
            
            self.session.add(db_prompt)
            self.session.commit()
            self.session.refresh(db_prompt)
            
            logger.info(f"Advertising prompt built: {db_prompt.id}")
            return db_prompt
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Prompt building failed: {str(e)}")
            raise ValueError(f"Prompt building failed: {str(e)}")


    async def generate_image(
        self, 
        prompt_id: int, 
        reference_image_bytes_list: list[bytes] | None = None, 
        session_id: str | None = None
    ) -> GeneratedImage:
        """
        Generate final ad image using prompt and input images.
        
        Args:
            prompt_id: ID of the advertising prompt
            reference_image_bytes_list: Optional reference images for generation
            session_id: Session identifier (optional, will use prompt's session_id if not provided)
            
        Returns:
            GeneratedImage: Database record with generated image
            
        Raises:
            ValueError: If image generation fails
            RuntimeError: If database operation fails
        """
        try:
            prompt = self.session.get(Prompt, prompt_id)
            if not prompt:
                raise ValueError(f"Prompt with ID {prompt_id} not found")
            
            # Get the product image from the database
            product_analysis = self.session.get(ImageAnalysis, prompt.image_analysis_id)
            if not product_analysis or not product_analysis.image_path:
                raise ValueError(f"Product image not found for prompt {prompt_id}")
            
            # Read product image from disk
            with open(product_analysis.image_path, "rb") as f:
                product_image_bytes = f.read()
            
            # Save reference images to disk if provided
            saved_reference_paths = []
            if reference_image_bytes_list:
                for idx, ref_bytes in enumerate(reference_image_bytes_list):
                    saved_ref = self._save_image(ref_bytes, "references", f"ref_{idx+1}")
                    saved_reference_paths.append(saved_ref)
            
            # Generate image using the dedicated image generator module
            # Note: API key already validated in __init__, so self.openai_api_key is guaranteed to exist
            data_url = await generate_image_data_url(
                prompt=prompt.prompt_text,
                product_image_bytes=product_image_bytes,
                model=self.img_model,
                api_key=self.openai_api_key,
                reference_images_bytes=reference_image_bytes_list,
            )

            # Persist generated image locally and expose it via /static
            os.makedirs("output_images", exist_ok=True)
            # Create a unique filename that associates the file with this session
            filename = f"generated_{session_id or 'session'}_{uuid.uuid4().hex}.png"
            output_path = os.path.join("output_images", filename)

            # Since generate_image_data_url always returns a data URL format,
            # we know it will be: "data:image/png;base64,<base64-bytes>"
            try:
                if not data_url or not data_url.startswith("data:image"):
                    raise ValueError("Invalid data URL format from image generator")
                
                # Extract the base64 data from the data URL
                # Format: "data:image/png;base64,ABC123..."
                header, b64data = data_url.split(",", 1)
                
                # Decode base64 to raw bytes and save to disk
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(b64data))
                
                # Return a stable local URL that won't expire
                final_local_url = f"/static/{filename}"
                
            except Exception as e:
                # If saving fails, log the error and return the data URL
                # (at least the image data is still accessible in the database)
                logger.error(f"Failed to save generated image locally: {str(e)}")
                final_local_url = data_url

            # Store the exact file paths used for generation
            used_paths = [product_analysis.image_path] if product_analysis.image_path else []
            used_paths.extend(saved_reference_paths)
            
            # Use provided session_id or get it from the prompt
            final_session_id = session_id or prompt.session_id
            
            db_ad_img = GeneratedImage(
                prompt_id=prompt_id,
                image_url=final_local_url,
                input_images=used_paths,
                session_id=final_session_id
            )
            
            self.session.add(db_ad_img)
            self.session.commit()
            self.session.refresh(db_ad_img)
            
            logger.info(f"Image generation completed: {db_ad_img.id}")
            return db_ad_img
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Image generation failed: {str(e)}")
            raise ValueError(f"Image generation failed: {str(e)}")


    async def create_final_prompt(
        self,
        product_image_bytes: bytes,
        user_vision_text: str,
        focus_slider: int,
        session_id: str,
        moodboard_image_bytes_list: list[bytes] | None = None
    ) -> Prompt:
        try:
            # Step 1: Analyze product image
            product_analysis = await self.analyze_product_image(product_image_bytes, session_id)
            
            # Step 2: Analyze moodboard images
            moodboard_ids: list[int] = []
            if moodboard_image_bytes_list:
                moodboard_analyses = await self.analyze_moodboard_images(session_id, moodboard_image_bytes_list)
                moodboard_ids = [analysis.id for analysis in moodboard_analyses]
            
            # Step 3: Parse user vision
            user_vision = await self.parse_user_vision(user_vision_text, session_id)
            
            # Step 4: Build advertising prompt
            prompt = await self.build_advertising_prompt(
                product_analysis.id,
                user_vision.id,
                focus_slider,
                session_id,
                moodboard_ids
            )

            return prompt

        except Exception as e:
            logger.error(f"Complete ad prompt generation workflow failed: {str(e)}")
            raise ValueError(f"Ad prompt generation workflow failed: {str(e)}")
    

    async def refine_prompt(
        self,
        previous_prompt_id: int,
        session_id: str,
        user_feedback: str | None = None,
        focus_slider: int | None = None
    ) -> Prompt:
        """
        Refine an existing prompt by generating an improved version.
        
        This method reuses all existing analysis data (product, moodboard, user vision)
        from the previous prompt, avoiding expensive re-analysis. It extracts the
        analysis IDs from the previous prompt and calls build_advertising_prompt
        with is_refinement=True, previous_prompt_text, and optional user_feedback
        to generate a better prompt.
        
        Why this approach provides:
        - Clean separation: refinement logic separate from core prompt building
        - Separation of concerns: delegates to build_advertising_prompt rather than duplicating logic
        - ID safety: prevents passing wrong analysis IDs by extracting from previous prompt
        - Consistency: calls build_advertising_prompt where validation, error handling, and database operations happen
        - Flexibility: enables focus_slider adjustments and user feedback without re-analysis
        
        Args:
            previous_prompt_id: ID of the prompt to refine
            session_id: Session identifier for linking the new prompt
            focus_slider: New focus level (0-10). If None, uses previous prompt's focus_slider
            user_feedback: Optional user feedback to guide the refinement
            
        Returns:
            Prompt: New refined prompt with incremented refinement_count
            
        Raises:
            ValueError: If previous prompt not found, focus_slider invalid, or max refinements exceeded
        """
        try:
            # Get previous prompt and validate it exists
            previous_prompt = self.session.get(Prompt, previous_prompt_id)
            if not previous_prompt:
                raise ValueError(f"Previous prompt with ID {previous_prompt_id} not found")
            
            # Use previous focus_slider if not provided
            final_focus_slider = focus_slider if focus_slider is not None else previous_prompt.focus_slider
            
            # Inside try-block: validation depends on database data (previous_prompt.focus_slider)
            # Must fetch previous_prompt first to calculate final_focus_slider before validating
            if not (0 <= final_focus_slider <= 10):
                raise ValueError("Focus slider must be between 0 and 10.")
            
            # Extract data from previous prompt (reuse existing analyses)
            image_analysis_id = previous_prompt.image_analysis_id
            user_vision_id = previous_prompt.user_vision_id
            moodboard_analysis_ids = previous_prompt.moodboard_analysis_ids

            # Call service method with refinement parameters
            refined_prompt = await self.build_advertising_prompt(
                image_analysis_id, 
                user_vision_id, 
                final_focus_slider, 
                session_id, 
                moodboard_analysis_ids, 
                is_refinement=True, 
                previous_prompt_id=previous_prompt_id, 
                user_feedback=user_feedback
            )

            return refined_prompt

        except Exception as e:
            self.session.rollback()
            logger.error(f"Prompt refinement failed: {str(e)}")
            raise ValueError(f"Prompt refinement failed: {str(e)}")


    async def create_complete_ad(
        self,
        product_image_bytes: bytes,
        user_vision_text: str,
        focus_slider: int,
        session_id: str,
        moodboard_image_bytes_list: list[bytes] | None = None,
        reference_image_bytes_list: list[bytes] | None = None,
    ) -> GeneratedImage:
        """
        Complete workflow: analyze product, analyze moodboard, parse vision,
        build prompt, and generate final ad image.
        
        Args:
            product_image_bytes: Product image data
            moodboard_image_bytes_list: List of moodboard image data
            user_vision_text: User's vision description
            focus_slider: Focus level (0-10)
            session_id: Session identifier for linking all records
            reference_image_bytes_list: Optional reference images for final generation
            
        Returns:
            GeneratedImage: Final generated ad image
            
        Raises:
            ValueError: If any step in the workflow fails
        """
        try:
            # Step 1: Analyze product image
            product_analysis = await self.analyze_product_image(product_image_bytes, session_id)
            
            # Step 2: Analyze moodboard images
            moodboard_ids: list[int] = []
            if moodboard_image_bytes_list:
                moodboard_analyses = await self.analyze_moodboard_images(session_id, moodboard_image_bytes_list)
                moodboard_ids = [analysis.id for analysis in moodboard_analyses]
            
            # Step 3: Parse user vision
            user_vision = await self.parse_user_vision(user_vision_text, session_id)
            
            # Step 4: Build advertising prompt
            prompt = await self.build_advertising_prompt(
                product_analysis.id,
                user_vision.id,
                focus_slider,
                session_id,
                moodboard_ids
            )
            
            # Step 5: Generate final image using product + optional references
            final_image = await self.generate_image(prompt.id, reference_image_bytes_list, session_id)
            
            logger.info(f"Complete ad generation workflow finished: {final_image.id}")
            return final_image
            
        except Exception as e:
            logger.error(f"Complete ad generation workflow failed: {str(e)}")
            raise ValueError(f"Ad generation workflow failed: {str(e)}")




