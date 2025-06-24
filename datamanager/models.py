from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class User(db.Model):

    __tablename__ = 'users'
    pass


class Image(db.Model):

    __tablename__ = 'images'
    pass


class Keyword(db.Model):

    __tablename__ = 'keywords'
    pass


class Hashtag(db.Model):

    __tablename__ = 'hashtags'
    pass


class Prompt(db.Model):

    __tablename__ = 'prompts'
    pass


class GeneratedImage(db.Model):

    __tablename__ = 'generated_images'
    pass


class Feedback(db.Model):

    __tablename__ = 'feedback'
    pass