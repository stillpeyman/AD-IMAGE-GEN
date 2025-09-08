# stdlib imports
import base64
import logging
import os
import uuid

# third-party imports
import requests
from sqlmodel import Session, select

# local imports
from agents import Agents
from api.image_generator import generate_image_data_url
from models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt, GeneratedImage


logger = logging.getLogger(__name__)


class AdGeneratorService:
    """
    Service layer for ad image generation workflow.
    
    Handles the complete process from image analysis to final ad generation,
    including database operations and error handling.
    """
    
    def __init__(self, agents: Agents, session: Session, img_api_key: str | None, img_model: str):
        """
        Initialize the service with agents, database session, and image generation config.
        
        Args:
            agents: The AI agents for text analysis tasks
            session: Database session for persistence
            img_api_key: API key for image generation (MY_OPENAI_API_KEY)
            img_model: Model name for image generation (e.g., "gpt-image-1")
        
        Attributes:
            session_memory: Dictionary storing session data for refinement logic
        """
        self.agents = agents
        self.session = session
        self.img_api_key = img_api_key
        self.img_model = img_model
        self.session_memory = {}


    def _store_session_data(self, session_id: str, step: str, data: dict) -> None:
        """
        Store analysis data in session memory for refinement logic.
        
        Private method - internal use only, not part of public API.
        Creates session structure if it doesn't exist, then stores step data.
        
        Args:
            session_id: Unique session identifier
            step: Analysis step name (e.g., "product_analysis", "moodboard_analysis")
            data: Dictionary containing id, analysis, and refinement_count
        """
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {}
        self.session_memory[session_id][step] = data


    def _get_session_data(self, session_id: str, step: str = None) -> dict | None:
        """
        Retrieve analysis data from session memory.
        
        Private method - internal use only, not part of public API.
        Can return entire session or specific step data.
        
        Args:
            session_id: Unique session identifier
            step: Analysis step name (optional, defaults to None)
                - None: returns entire session data
                - "product_analysis": returns only that step's data
        
        Returns:
            dict | None: Session data, step data, or None if not found
        """
        if session_id not in self.session_memory:
            return None
        
        if step is None:
            return self.session_memory[session_id]

        # .get(step) returns None if step doesn't exist
        return self.session_memory[session_id].get(step)


    # static methods = utility functions
    # don't need instance data, work independently of the class, don't access <self>
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


    async def analyze_product_image(self, image_bytes: bytes, session_id: str, refinement_feedback: str = None) -> ImageAnalysis:
        """
        Analyze a product image and store results in database.

        Args:
            image_bytes: Raw product image data
            session_id: Session identifier for linking related records
            refinement_feedback: Optional feedback for refinement (defaults to None)
            
        Returns:
            ImageAnalysis: Database record with analysis results
            
        Raises:
            ValueError: If image analysis fails
            RuntimeError: If database operation fails
        """
        try:
            if refinement_feedback:
                # Get original image path from session memory
                session_data = self._get_session_data(session_id, "product_analysis")
                image_path = session_data["analysis"].image_path

                # Read original image from disk for refinement analysis
                with open(image_path, "rb") as f:
                    original_image_bytes = f.read()

                analysis = await self.agents.analyze_product_image(original_image_bytes, refinement_feedback)

            else:
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

            self._store_session_data(
                session_id, 
                "product_analysis", 
                {
                "id": db_analysis.id,
                "analysis": db_analysis,
                "refinement_count": 0 if not refinement_feedback else session_data["refinement_count"] + 1
                }
                )
            
            logger.info(f"Product image analysis completed: {db_analysis.id}")
            return db_analysis
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Product image analysis failed: {str(e)}")
            raise ValueError(f"Product image analysis failed: {str(e)}")


    async def analyze_moodboard_images(self, image_bytes_list: list[bytes] | None, session_id: str) -> list[MoodboardAnalysis]:
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


    async def parse_user_vision(self, user_text: str | None, session_id: str) -> UserVision | None:
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
            return None
        
        try:
            analysis = await self.agents.parse_user_vision(user_text)
            
            db_analysis = UserVision(
                subjects=analysis.subjects,
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
        image_analysis_id: int,
        moodboard_analysis_ids: list[int],
        user_vision_id: int | None,
        focus_slider: int,
        session_id: str
    ) -> Prompt:
        """
        Build advertising prompt using analysis results and store in database.
        
        Args:
            image_analysis_id: ID of product image analysis
            moodboard_analysis_ids: List of moodboard analysis IDs
            user_vision_id: ID of user vision analysis
            focus_slider: Focus level (0-10)
            session_id: Session identifier for linking related records
            
        Returns:
            Prompt: Database record with generated prompt
            
        Raises:
            ValueError: If prompt building fails
            RuntimeError: If database operation fails
        """
        if not (0 <= focus_slider <= 10):
            raise ValueError("Focus slider must be between 0 and 10")
        
        try:
            # Get all required analyses
            image_analysis = self.session.get(ImageAnalysis, image_analysis_id)
            if not image_analysis:
                raise ValueError(f"Image analysis with ID {image_analysis_id} not found")
            
            moodboard_analyses = []
            for moodboard_id in moodboard_analysis_ids:
                analysis = self.session.get(MoodboardAnalysis, moodboard_id)
                if not analysis:
                    raise ValueError(f"Moodboard analysis with ID {moodboard_id} not found")
                moodboard_analyses.append(analysis)
            
            user_vision = None
            if user_vision_id is not None:
                user_vision = self.session.get(UserVision, user_vision_id)
                if not user_vision:
                    raise ValueError(f"User vision with ID {user_vision_id} not found")
            
            # Use all moodboard analyses for prompt building
            prompt = await self.agents.build_advertising_prompt(
                image_analysis, 
                moodboard_analyses,  # Pass all moodboard analyses
                user_vision, 
                focus_slider
            )
            
            db_prompt = Prompt(
                prompt_text=prompt.prompt_text,
                image_analysis_id=image_analysis_id,
                moodboard_analysis_ids=moodboard_analysis_ids,  # Store all moodboard IDs
                user_vision_id=user_vision_id,
                focus_slider=focus_slider,
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


    async def generate_image(self, prompt_id: int, reference_image_bytes_list: list[bytes] | None = None, session_id: str | None = None) -> GeneratedImage:
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
            if not self.img_api_key:
                raise ValueError("Image generation API key not configured. Please set MY_OPENAI_API_KEY environment variable.")
            
            data_url = await generate_image_data_url(
                prompt=prompt.prompt_text,
                product_image_bytes=product_image_bytes,
                reference_images_bytes=reference_image_bytes_list,
                model=self.img_model,
                api_key=self.img_api_key,
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


    async def create_complete_ad(
        self,
        product_image_bytes: bytes,
        moodboard_image_bytes_list: list[bytes] | None,
        user_vision_text: str | None,
        focus_slider: int,
        session_id: str,
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
                moodboard_analyses = await self.analyze_moodboard_images(moodboard_image_bytes_list, session_id)
                moodboard_ids = [analysis.id for analysis in moodboard_analyses]
            
            # Step 3: Parse user vision (optional)
            user_vision = await self.parse_user_vision(user_vision_text, session_id)
            
            # Step 4: Build advertising prompt
            prompt = await self.build_advertising_prompt(
                product_analysis.id,
                moodboard_ids,
                (user_vision.id if user_vision else None),
                focus_slider,
                session_id
            )
            
            # Step 5: Generate final image using product + optional references
            final_image = await self.generate_image(prompt.id, reference_image_bytes_list, session_id)
            
            logger.info(f"Complete ad generation workflow finished: {final_image.id}")
            return final_image
            
        except Exception as e:
            logger.error(f"Complete ad generation workflow failed: {str(e)}")
            raise ValueError(f"Ad generation workflow failed: {str(e)}")




