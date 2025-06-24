from .data_manager_interface import DataManagerInterface
from .models import db, User, Image, Keyword, Hashtag, Prompt, GeneratedImage, Feedback


class SQLiteDataManager(DataManagerInterface):
    
    def create_user(self, email: str, password: str):
        pass
    
    def get_user_by_id(self, user_id: str):
        pass

    def save_uploaded_image(self, user_id: str, filename: str):
        pass

    def get_image_by_id(self, image_id: str):
        pass

    def delete_image(self, image_id: str):
        pass

    def add_keywords_for_image(self, image_id: str, keywords: list[str]):
        pass

    def get_keywords_for_image(self, image_id: str) -> list[str]:
        pass

    def get_trending_hashtags(self, keyword: str):
        pass

    def generate_prompts_for_image(self, image_id: str, model_names: list[str]):
        pass

    def get_prompt_by_id(self, prompt_id: str):
        pass

    def approve_prompt(self, prompt_id: str):
        pass

    def get_generated_image_by_id(self, generated_image_id: str):
        pass
    
    # keep in mind: needs change if feedback on generated image
    def save_feedback(self, prompt_id: str, rating: int, comment: str):
        pass

    def get_user_history(self, user_id: str):
        pass