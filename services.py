from sqlmodel import Session, select
from .agents import Agents
from .models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt, GeneratedImage
import base64
from typing import Optional
import logging
import os
import uuid


logger = logging.getLogger(__name__)


class AdGeneratorService:
    """
    Service layer for ad image generation workflow.
    
    Handles the complete process from image analysis to final ad generation,
    including database operations and error handling.
    """
    
    def __init__(self, agents: Agents, session: Session):
        """
        Initialize the service with agents and database session.
        
        Args:
            agents: The AI agents for image analysis and generation
            session: Database session for persistence
        """
        self.agents = agents
        self.session = session


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


    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        """
        Encode image bytes to base64 string for API input.
        
        Args:
            image_bytes: Raw image data
            
        Returns:
            Base64 encoded string
            
        Raises:
            ValueError: If image_bytes is empty or invalid
        """
        if not image_bytes:
            raise ValueError("Image bytes cannot be empty")
        
        try:
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to encode image: {str(e)}")


    async def analyze_product_image(self, image_bytes: bytes) -> ImageAnalysis:
        """
        Analyze a product image and store results in database.
        
        Args:
            image_bytes: Raw product image data
            
        Returns:
            ImageAnalysis: Database record with analysis results
            
        Raises:
            ValueError: If image analysis fails
            RuntimeError: If database operation fails
        """
        try:
            image_path = self._save_image(image_bytes, "product", "product")
            base64_image = self.encode_image(image_bytes)
            analysis = await self.agents.analyze_product_image(base64_image)
            
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
                image_path=image_path
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


    async def analyze_moodboard_images(self, image_bytes_list: list[bytes]) -> list[MoodboardAnalysis]:
        """
        Analyze multiple moodboard images and store results in database.
        
        Args:
            image_bytes_list: List of raw moodboard image data
            
        Returns:
            List[MoodboardAnalysis]: Database records with analysis results
            
        Raises:
            ValueError: If moodboard analysis fails
            RuntimeError: If database operation fails
        """
        if not image_bytes_list:
            raise ValueError("Moodboard image list cannot be empty")
        
        results = []
        
        try:
            # Save all moodboard images and convert to base64
            base64_images = []
            saved_paths = []
            for idx, image_bytes in enumerate(image_bytes_list):
                saved_path = self._save_image(image_bytes, "moodboards", f"moodboard_{idx+1}")
                saved_paths.append(saved_path)
                base64_images.append(self.encode_image(image_bytes))
            
            # Analyze all moodboard images at once
            analyses = await self.agents.analyze_moodboard(base64_images)
            
            # Create database records for each analysis with corresponding image_path
            for analysis, saved_path in zip(analyses, saved_paths):
                db_analysis = MoodboardAnalysis(
                    scene_description=analysis.scene_description,
                    visual_style=analysis.visual_style,
                    mood_atmosphere=analysis.mood_atmosphere,
                    color_theme=analysis.color_theme,
                    composition_patterns=analysis.composition_patterns,
                    suggested_keywords=analysis.suggested_keywords,
                    image_path=saved_path
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


    async def parse_user_vision(self, user_text: str) -> UserVision:
        """
        Parse user vision text and store results in database.
        
        Args:
            user_text: User's vision description
            
        Returns:
            UserVision: Database record with parsed vision
            
        Raises:
            ValueError: If user vision parsing fails
            RuntimeError: If database operation fails
        """
        if not user_text or not user_text.strip():
            raise ValueError("User vision text cannot be empty")
        
        try:
            analysis = await self.agents.parse_user_vision(user_text)
            
            db_analysis = UserVision(
                subjects=analysis.subjects,
                action=analysis.action,
                setting=analysis.setting,
                lighting=analysis.lighting,
                mood_descriptors=analysis.mood_descriptors,
                additional_details=analysis.additional_details
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
        user_vision_id: int,
        focus_slider: int
    ) -> Prompt:
        """
        Build advertising prompt using analysis results and store in database.
        
        Args:
            image_analysis_id: ID of product image analysis
            moodboard_analysis_ids: List of moodboard analysis IDs
            user_vision_id: ID of user vision analysis
            focus_slider: Focus level (0-10)
            
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
                focus_slider=focus_slider
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


    async def generate_image(self, prompt_id: int, reference_image_bytes_list: list[bytes] | None = None) -> GeneratedImage:
        """
        Generate final ad image using prompt and input images.
        
        Args:
            prompt_id: ID of the advertising prompt
            reference_image_bytes_list: Optional reference images for generation
            
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
            
            # Encode product image
            product_image_base64 = self.encode_image(product_image_bytes)
            
            # Encode reference images if provided
            reference_images_base64 = []
            if reference_image_bytes_list:
                for image_bytes in reference_image_bytes_list:
                    base64_image = self.encode_image(image_bytes)
                    reference_images_base64.append(base64_image)
            
            # Save reference images to disk if provided
            saved_reference_paths = []
            if reference_image_bytes_list:
                for idx, ref_bytes in enumerate(reference_image_bytes_list):
                    saved_ref = self._save_image(ref_bytes, "references", f"ref_{idx+1}")
                    saved_reference_paths.append(saved_ref)
            
            final_image = await self.agents.generate_image(
                prompt.prompt_text, 
                product_image_base64, 
                reference_images_base64 if reference_images_base64 else None
            )

            # Store the exact file paths used for generation
            used_paths = [product_analysis.image_path] if product_analysis.image_path else []
            used_paths.extend(saved_reference_paths)
            
            db_ad_img = GeneratedImage(
                prompt_id=prompt_id,
                image_url=final_image.image_url,
                input_images=used_paths
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
        moodboard_image_bytes_list: list[bytes],
        user_vision_text: str,
        focus_slider: int,
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
            reference_image_bytes_list: Optional reference images for final generation
            
        Returns:
            GeneratedImage: Final generated ad image
            
        Raises:
            ValueError: If any step in the workflow fails
        """
        try:
            # Step 1: Analyze product image
            product_analysis = await self.analyze_product_image(product_image_bytes)
            
            # Step 2: Analyze moodboard images
            moodboard_analyses = await self.analyze_moodboard_images(moodboard_image_bytes_list)
            moodboard_ids = [analysis.id for analysis in moodboard_analyses]
            
            # Step 3: Parse user vision
            user_vision = await self.parse_user_vision(user_vision_text)
            
            # Step 4: Build advertising prompt
            prompt = await self.build_advertising_prompt(
                product_analysis.id,
                moodboard_ids,
                user_vision.id,
                focus_slider
            )
            
            # Step 5: Generate final image using product + optional references
            final_image = await self.generate_image(prompt.id, reference_image_bytes_list)
            
            logger.info(f"Complete ad generation workflow finished: {final_image.id}")
            return final_image
            
        except Exception as e:
            logger.error(f"Complete ad generation workflow failed: {str(e)}")
            raise ValueError(f"Ad generation workflow failed: {str(e)}")




