from abc import ABC, abstractmethod


class DataManagerInterface(ABC):

    @abstractmethod
    def create_user(self, email: str, password: str):
        pass
    
    # Later, if using FastAPI, may convert to user_id type to UUID
    @abstractmethod
    def get_user_by_id(self, user_id: str):
        pass

    @abstractmethod
    def save_uploaded_image(self, user_id: str, filename: str):
        pass

    @abstractmethod
    def get_image_by_id(self, image_id: str):
        pass

    @abstractmethod
    def delete_image(self, image_id: str):
        pass

    @abstractmethod
    def add_keywords_for_image(self, image_id: str, keywords: list[str]):
        pass

    @abstractmethod
    def get_keywords_for_image(self, image_id: str) -> list[str]:
        pass

    @abstractmethod
    def get_trending_hashtags(self, keyword: str):
        pass

    @abstractmethod
    def generate_prompts_for_image(self, image_id: str, model_names: list[str]):
        pass

    @abstractmethod
    def get_prompt_by_id(self, prompt_id: str):
        pass

    @abstractmethod
    def approve_prompt(self, prompt_id: str):
        pass

    @abstractmethod
    def get_generated_image_by_id(self, generated_image_id: str):
        pass
    
    # keep in mind: needs change if feedback on generated image
    @abstractmethod
    def save_feedback(self, prompt_id: str, rating: int, comment: str):
        pass

    @abstractmethod
    def get_user_history(self, user_id: str):
        pass




