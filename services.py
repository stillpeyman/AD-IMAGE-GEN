# stdlib imports
import base64
import logging
import os
import uuid
import threading

# third-party imports
from sqlmodel import Session, select
from sqlalchemy import func

# local imports
from agents import Agents
from api.gpt_image1_generator import generate_image_data_url as gpt_generate_image_data_url
from api.gemini_image_generator import generate_image_data_url as gemini_generate_image_data_url
from models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt, GeneratedImage, PromptExample


logger = logging.getLogger(__name__)


def _log_sess(msg, session_obj):
    """
    Debug helper to log thread ID and session object ID.
    
    Used to track if requests switch threads during async operations
    and to verify session object identity remains consistent.
    
    Args:
        msg: Descriptive message about what's happening
        session_obj: The SQLAlchemy session object to log
    
    Logs format: [tid=12345] message session_obj_id=140234567890
    """
    logger.info(f"[tid={threading.get_ident()}] {msg} session_obj_id={id(session_obj)}")


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
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None
    ):
        """
        Initialize the service with agents, database session, and image generation config.
        
        Args:
            agents: The AI agents for text analysis tasks (uses 'gpt-4.1' or
                    'gemini-2.5-flash' depending on the session's provider)
            session: Database session for persistence
            openai_api_key: OpenAI API key for image generation (optional; required when image_model_choice="openai").
            gemini_api_key: Google API key for Gemini image generation (optional; required when image_model_choice="google").
        
        Attributes:
            agents: AI agents configured with the session's model provider
            session: Database session for persistence
            openai_api_key: OpenAI API key for image generation
            gemini_api_key: Google API key for Gemini image generation
        """
        # Store API keys and configuration
        # Note: Text analysis validation happens in Agents.__init__()
        # Image generation validation happens at point of use in generate_image()
        self.agents = agents
        self.session = session
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key


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


    async def analyze_product_image(self, image_bytes: bytes, user_session_id: str) -> ImageAnalysis:
        """
        Analyze a product image and store results.

        Uses self.agents for analysis and persists the structured result linked to the user session.

        Args:
            image_bytes: Raw product image data.
            user_session_id: User session identifier to link created records.

        Returns:
            Persisted ImageAnalysis.

        Raises:
            ValueError: If analysis or persistence fails.
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
                session_id=user_session_id,
                model_provider=self.agents.model_provider  
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
        user_session_id: str, 
        image_bytes_list: list[bytes] | None = None
    ) -> list[MoodboardAnalysis]:
        """
        Analyze moodboard images (if provided) and store results.

        Args:
            image_bytes_list: Moodboard images to analyze; if None/empty, returns [].
            user_session_id: User session identifier to link created records.

        Returns:
            List of persisted MoodboardAnalysis.

        Raises:
            ValueError: If analysis or persistence fails.
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
                    session_id=user_session_id,
                    model_provider=self.agents.model_provider  
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


    async def parse_user_vision(self, user_text: str, user_session_id: str) -> UserVision:
        """
        Parse user vision text and store structured results.

        Args:
            user_text: User's vision description.
            user_session_id: User session identifier to link created records.

        Returns:
            Persisted UserVision.

        Raises:
            ValueError: If parsing or persistence fails.
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
                session_id=user_session_id,
                model_provider=self.agents.model_provider 
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
        user_session_id: str,
        # optional params last
        moodboard_analysis_ids: list[int] | None = None,
        is_refinement: bool = False,
        previous_prompt_id: int | None = None,
        user_feedback: str | None = None
    ) -> Prompt:
        """
        Build and persist an advertising prompt using prior analyses.

        Optionally retrieves two category-matched PromptExample rows for Few-Shot prompting.

        Args:
            image_analysis_id: Product image analysis to use.
            user_vision_id: Parsed user vision to use.
            focus_slider: Balance (0–10) between product and scene.
            user_session_id: User session identifier to link the prompt.
            moodboard_analysis_ids: Optional moodboard analysis IDs.
            is_refinement: Whether this is a refinement iteration.
            previous_prompt_id: Previous prompt ID for refinement context.
            user_feedback: Optional feedback to steer refinement.

        Returns:
            Persisted Prompt.

        Raises:
            ValueError: If required analyses are missing or generation fails.
        """
        try:
            # Get all required analyses, if not found SQLModel returns None 
            # All db objects bound to session
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
            prompt_examples_db = self.session.exec(stmt).all()
            
            # Convert to plain dicts to avoid SQLAlchemy lazy-loading issues
            # Extract all data from DB objects while session is active
            prompt_examples = [
                {
                    "prompt_text": ex.prompt_text,
                    "product_category": ex.product_category
                }
                for ex in prompt_examples_db
            ]

            # Commit and close to release database lock before long AI call
            self.session.commit()
            engine = self.session.get_bind()
            self.session.close()

            # Make AI call (long-running operation, no database lock held)
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
            
            # Save results with a fresh session
            db_prompt = Prompt(
                prompt_text=prompt.prompt_text,
                image_analysis_id=image_analysis_id,
                moodboard_analysis_ids=moodboard_analysis_ids or [],
                user_vision_id=user_vision_id,
                focus_slider=focus_slider,
                refinement_count=refinement_count,
                user_feedback=user_feedback,
                previous_prompt_id=previous_prompt_id,
                session_id=user_session_id,
                model_provider=self.agents.model_provider 
            )
            
            with Session(engine) as write_session:
                write_session.add(db_prompt)
                write_session.commit()
                write_session.refresh(db_prompt)
            
            logger.info(f"Advertising prompt built: {db_prompt.id}")
            return db_prompt
            
        except Exception as e:
            try:
                self.session.rollback()
                self.session.close()
            except Exception:
                pass
            logger.error(f"Prompt building failed: {str(e)}")
            raise ValueError(f"Prompt building failed: {str(e)}")
    

    async def refine_prompt(
        self,
        previous_prompt_id: int,
        user_session_id: str,
        user_feedback: str | None = None,
        focus_slider: int | None = None
    ) -> Prompt:
        """
        Refine a saved prompt by generating a new iteration.

        Reuses prior analyses (product, moodboard, user vision) referenced by the previous prompt to avoid re-analysis. Delegates to build_advertising_prompt.

        Args:
            previous_prompt_id: Prompt to refine.
            user_session_id: User session identifier to link the refined prompt.
            focus_slider: Optional new focus value; defaults to previous value when None.
            user_feedback: Optional feedback text to guide refinement.

        Returns:
            Persisted refined Prompt.

        Raises:
            ValueError: If previous prompt is missing, focus invalid, or max refinements exceeded.
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

            # Delegate to build_advertising_prompt which handles its own session management
            # (loads data → commits → closes → AI call → new session → saves)
            refined_prompt = await self.build_advertising_prompt(
                image_analysis_id, 
                user_vision_id, 
                final_focus_slider, 
                user_session_id, 
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


    async def generate_image(
        self, 
        prompt_id: int, 
        image_model_choice: str,
        user_session_id: str,
        reference_image_bytes_list: list[bytes] | None = None
    ) -> GeneratedImage:
        """
        Generate and persist the final ad image using a saved prompt.
        
        IMPORTANT: This method prevents SQLite "database is locked" errors by:
        1. Loading all data and committing the read transaction BEFORE the API call
        2. Making the long-running API call with no database lock held
        3. Opening a fresh transaction to save the results
        
        This prevents timeout issues when image generation takes >5 seconds.

        Args:
            prompt_id: Prompt to render.
            image_model_choice: Image generation model choice ("openai" or "google").
            user_session_id: User session identifier to link the generated image.
            reference_image_bytes_list: Optional reference images.

        Returns:
            Persisted GeneratedImage (with local /static URL when possible).

        Raises:
            ValueError: If generation or persistence fails.
        """
        try:
            # PHASE 1: Load all needed data from database
            prompt = self.session.get(Prompt, prompt_id)
            if not prompt:
                raise ValueError(f"Prompt with ID {prompt_id} not found")
            
            product_analysis = self.session.get(ImageAnalysis, prompt.image_analysis_id)
            if not product_analysis or not product_analysis.image_path:
                raise ValueError(f"Product image not found for prompt {prompt_id}")
            
            # Extract all data (primitive data -> strings, paths) 
            # as Python variables (no longer tied to DB session)
            prompt_text = prompt.prompt_text
            product_image_path = product_analysis.image_path
            
            # CRITICAL FIX: Commit and close to release database lock before long API call
            # -------------------------------------------------------------------------
            # This prevents "database is locked" errors when Gemini API takes 10+ seconds
            # Pattern: Load data → Commit → Close → Long operation → New session → Save  
            # Why this is necessary:
            # - SQLite allows only one writer at a time
            # - Holding a transaction during the 10+ second Gemini call blocks other requests
            # - WAL mode + check_same_thread=False help, but we still need to release the lock
            # - Solution: Commit and close before the long operation, create new session after
            
            _log_sess("before commit (phase1)", self.session)
            # Save transaction, release write lock
            # (only read data -> still a transaction)
            self.session.commit()  
            _log_sess("after commit (phase1)", self.session)
            
            # Get engine reference before closing session (needed to create new session later)
            # session.get_bind() returns the Engine object that this session is using
            engine = self.session.get_bind()
            
            # Close session to fully release the database connection back to the pool
            # This makes the connection available for other requests during the long Gemini call
            self.session.close()
            _log_sess("closed request session", self.session)
            
            # PHASE 2: File operations and API call (no database lock held)
            with open(product_image_path, "rb") as f:
                product_image_bytes = f.read()
            
            # Save reference images to disk if provided
            saved_reference_paths = []
            if reference_image_bytes_list:
                for idx, ref_bytes in enumerate(reference_image_bytes_list):
                    saved_ref = self._save_image(ref_bytes, "references", f"ref_{idx+1}")
                    saved_reference_paths.append(saved_ref)
            
            # Generate image using the user's chosen image model (independent of text analysis provider)
            # Validate API key availability for the chosen image model
            if image_model_choice == "openai":
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key is required for OpenAI image generation. Please set MY_OPENAI_API_KEY environment variable.")
                # Use GPT image generator
                data_url = await gpt_generate_image_data_url(
                    prompt=prompt_text,
                    product_image_bytes=product_image_bytes,
                    model="gpt-4.1",
                    api_key=self.openai_api_key,
                    reference_images_bytes=reference_image_bytes_list,
                )
            elif image_model_choice == "google":
                if not self.gemini_api_key:
                    raise ValueError("Gemini API key is required for Gemini image generation. Please set GEMINI_API_KEY environment variable.")
                # Use Gemini image generator
                data_url = await gemini_generate_image_data_url(
                    prompt=prompt_text,
                    product_image_bytes=product_image_bytes,
                    model="gemini-2.5-flash-image",
                    api_key=self.gemini_api_key,
                    reference_images_bytes=reference_image_bytes_list,
                )
            else:
                raise ValueError(f"Unsupported image model choice: {image_model_choice}. Must be 'openai' or 'google'.")

            # Persist generated image locally and expose it via /static
            os.makedirs("output_images", exist_ok=True)
            # Create a unique filename that associates the file with this session
            filename = f"generated_{user_session_id}_{uuid.uuid4().hex}.png"
            # Storage: File saved in output_images/ folder
            output_path = os.path.join("output_images", filename)

            # Since generate_image_data_url always returns a data URL format,
            # We know it will be: "data:image/png;base64,<base64-bytes>"
            # Data URL format benefits:
            # - Universal web compatibility (works in HTML <img src="...">)
            # - No temporary file management needed
            # - Consistent format between OpenAI and Google providers
            # - Can be saved to disk or displayed directly
            try:
                if not data_url or not data_url.startswith("data:image"):
                    raise ValueError("Invalid data URL format from image generator")
                
                # Extract the base64 data from the data URL
                # Format: "data:image/png;base64,ABC123..."
                header, b64data = data_url.split(",", 1)
                
                # Decode base64 to raw bytes and save to disk
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(b64data))
                
                # Serving: FastAPI serves file via /static/ route
                # Return a stable local URL that won't expire
                final_local_url = f"/static/{filename}"
                
            except Exception as e:
                # If saving fails, log the error and return the data URL
                # (at least the image data is still accessible in the database)
                logger.error(f"Failed to save generated image locally: {str(e)}")
                final_local_url = data_url

            # Store the exact file paths used for generation
            # product_image_path is guaranteed to exist (validated at line 386-387)
            used_paths = [product_image_path]
            used_paths.extend(saved_reference_paths)
            
            # PHASE 3: Save results to database with a fresh session
            # -------------------------------------------------------
            # We closed self.session earlier (line 429), so we need a new session to save results
            # Why a new session:
            # - self.session is closed and cannot be used anymore
            # - Creating a fresh session gets a new connection from the pool
            # - No risk of "database is locked" because we're not reusing the old session
            # - The engine we extracted at line 425 is used to create this new session
            
            db_ad_img = GeneratedImage(
                prompt_id=prompt_id,
                image_url=final_local_url,
                input_images=used_paths,
                session_id=user_session_id,
                model_provider=image_model_choice
            )
            
            # Create fresh session from engine (extracted at line 425 before closing old session)
            # Context manager (with-block) ensures session is properly closed after use
            with Session(engine) as write_session:
                try:
                    _log_sess("new write session created", write_session)
                    write_session.add(db_ad_img)     
                    write_session.commit()
                    write_session.refresh(db_ad_img)  
                    _log_sess("write session committed", write_session)
                except Exception:
                    write_session.rollback()  # Undo changes if error
                    raise  # Re-raise the error to be caught by outer except
            
            logger.info(f"Image generation completed: {db_ad_img.id}")
            return db_ad_img
            
        except Exception as e:
            # Outer exception handler: catches ANY error from the entire function
            # This could be: file not found, API failure, database error, etc.
            
            # Try to clean up the original request session (self.session)
            # Why nested try/except:
            # - If error happened after we closed the session (line 429), attempting
            #   rollback/close will fail because the session is already closed
            # - We don't care about cleanup errors, we care about the ORIGINAL error
            # - The inner try/except prevents cleanup errors from hiding the original error
            try:
                self.session.rollback()  # Attempt to undo any uncommitted changes
                self.session.close()     # Attempt to close the session
            except Exception:
                # If cleanup fails (e.g., session already closed), ignore it
                # We want to report the original error (e), not the cleanup error
                pass
            
            # Log and re-raise the ORIGINAL error (stored in variable 'e')
            logger.error(f"Image generation failed: {str(e)}")
            raise ValueError(f"Image generation failed: {str(e)}")


    async def create_final_prompt(
        self,
        product_image_bytes: bytes,
        user_vision_text: str,
        focus_slider: int,
        user_session_id: str,
        moodboard_image_bytes_list: list[bytes] | None = None
    ) -> Prompt:
        """
        Build a prompt in one call: analyze product, optionally analyze moodboards,
        parse user vision, then synthesize and persist the advertising prompt.

        The steps performed are equivalent to calling, in sequence:
        - analyze_product_image
        - analyze_moodboard_images (when moodboard images are provided)
        - parse_user_vision
        - build_advertising_prompt

        Args:
            product_image_bytes: Product image data to analyze
            user_vision_text: User's vision description to parse
            focus_slider: Focus balance (0-10) for product vs. scene
            user_session_id: User session identifier linking all created records
            moodboard_image_bytes_list: Optional moodboard images for additional context

        Returns:
            Prompt: Persisted prompt record ready for image generation

        Raises:
            ValueError: If any sub-step fails (analysis, parsing, or persistence)
        """
        try:
            # Step 1: Analyze product image
            product_analysis = await self.analyze_product_image(product_image_bytes, user_session_id)
            
            # Step 2: Analyze moodboard images
            moodboard_ids: list[int] = []
            if moodboard_image_bytes_list:
                moodboard_analyses = await self.analyze_moodboard_images(user_session_id, moodboard_image_bytes_list)
                moodboard_ids = [analysis.id for analysis in moodboard_analyses]
            
            # Step 3: Parse user vision
            user_vision = await self.parse_user_vision(user_vision_text, user_session_id)
            
            # Step 4: Build advertising prompt
            prompt = await self.build_advertising_prompt(
                product_analysis.id,
                user_vision.id,
                focus_slider,
                user_session_id,
                moodboard_ids
            )

            return prompt

        except Exception as e:
            logger.error(f"Complete ad prompt generation workflow failed: {str(e)}")
            raise ValueError(f"Ad prompt generation workflow failed: {str(e)}")


    async def create_complete_ad(
        self,
        product_image_bytes: bytes,
        user_vision_text: str,
        focus_slider: int,
        user_session_id: str,
        image_model_choice: str,
        moodboard_image_bytes_list: list[bytes] | None = None,
        reference_image_bytes_list: list[bytes] | None = None,
    ) -> GeneratedImage:
        """
        Full workflow: analyze product, optional moodboards, parse vision,
        build prompt, then generate the final ad image.

        Args:
            product_image_bytes: Product image to analyze.
            user_vision_text: User's scene/brand intent.
            focus_slider: Balance between product and scene (0–10).
            user_session_id: User session identifier to link all records.
            image_model_choice: Image generation model choice ("openai" or "google").
            moodboard_image_bytes_list: Optional moodboard images.
            reference_image_bytes_list: Optional reference images for final generation.

        Returns:
            Persisted GeneratedImage.

        Raises:
            ValueError: If any step in the workflow fails.
        """
        try:
            # Step 1: Analyze product image
            product_analysis = await self.analyze_product_image(product_image_bytes, user_session_id)
            
            # Step 2: Analyze moodboard images
            moodboard_ids: list[int] = []
            if moodboard_image_bytes_list:
                moodboard_analyses = await self.analyze_moodboard_images(user_session_id, moodboard_image_bytes_list)
                moodboard_ids = [analysis.id for analysis in moodboard_analyses]
            
            # Step 3: Parse user vision
            user_vision = await self.parse_user_vision(user_vision_text, user_session_id)
            
            # Step 4: Build advertising prompt
            prompt = await self.build_advertising_prompt(
                product_analysis.id,
                user_vision.id,
                focus_slider,
                user_session_id,
                moodboard_ids
            )
            
            # Step 5: Generate final image using product + optional references
            final_image = await self.generate_image(prompt.id, image_model_choice, user_session_id, reference_image_bytes_list)
            
            logger.info(f"Complete ad generation workflow finished: {final_image.id}")
            return final_image
            
        except Exception as e:
            logger.error(f"Complete ad generation workflow failed: {str(e)}")
            raise ValueError(f"Ad generation workflow failed: {str(e)}")


    async def save_prompt_example(self, prompt_id: int, user_session_id: str) -> PromptExample:
        """
        Persist a PromptExample derived from an existing Prompt.

        Workflow:
            - Load the Prompt by ID
            - Validate the Prompt belongs to the provided user_session_id
            - Load linked ImageAnalysis to obtain product_category
            - Create and persist PromptExample with prompt_id, prompt_text, product_category

        Args:
            prompt_id: ID of the Prompt to save as an example.
            user_session_id: The current workflow session ID; must match the Prompt.session_id.

        Returns:
            PromptExample: Newly persisted example row used by RAG retrieval.

        Raises:
            ValueError: If the Prompt does not exist, session mismatch occurs, or
                        the linked ImageAnalysis is missing.
        """
        prompt = self.session.get(Prompt, prompt_id)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")

        # Stricter safety: ensure the prompt belongs to the active user session
        if prompt.session_id != user_session_id:
            raise ValueError("Session mismatch: prompt does not belong to this user_session_id")

        image_analysis = self.session.get(ImageAnalysis, prompt.image_analysis_id)
        if not image_analysis:
            raise ValueError(
                f"ImageAnalysis {prompt.image_analysis_id} not found for prompt {prompt_id}"
            )

        prompt_example = PromptExample(
            prompt_id=prompt.id,
            prompt_text=prompt.prompt_text,
            product_category=image_analysis.product_category,
        )

        self.session.add(prompt_example)
        self.session.commit()
        self.session.refresh(prompt_example)

        return prompt_example



