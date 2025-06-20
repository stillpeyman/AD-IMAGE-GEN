from flask import Flask, request, jsonify


app = Flask(__name__)


# AUTHENTICATION
@app.route('/auth/register', methods=['POST'])
def register():
    return jsonify({'message': 'Register endpoint (stub)'}), 201


@app.route('/auth/login', methods=['POST'])
def login():
    return jsonify({'message': 'Login endpoint (stub)'}), 200


@app.route('/users/me', methods=['GET'])
def get_user_info():
    return jsonify({'message': 'User info endpoint (stub)'}), 200


# IMAGE UPLOAD & ANALYSIS
@app.route('/images/upload', methods=['POST'])
def upload_image():
    return jsonify({'message': 'Upload image endpoint (stub)'}), 200


@app.route('/images/<image_id>', methods=['GET'])
def get_image(image_id):
    return jsonify({'message': f'Get image {image_id} endpoint (stub)'}), 200

@app.route('/images/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    return jsonify({'message': f'Delete image {image_id} endpoint (stub)'}), 200


# TRENDING HASHTAGS
@app.route('/keywords/<keyword>/hashtags', methods=['GET'])
def fetch_hashtags(keyword):
    return jsonify({'message': f'Fetch hashtags for {keyword} endpoint (stub)'}), 200


# PROMPTS & PROMPT GENERATION
@app.route('/images/<image_id>/prompts', methods=['GET'])
def get_prompts(image_id):
    return jsonify({'message': f'Get prompts for {image_id} endpoint (stub)'}), 200


@app.route('/images/<image_id>/prompts/generate', methods=['POST'])
def generate_prompts(image_id):
    data = request.get_json()
    model_names = data.get('model_names', []) if data else []

    return jsonify({
        'message': 
        f'Generate prompts for {image_id} using models: {model_names} (stub)'}
        ), 200


# PROMPT APPROVAL & IMAGE GENERATION
@app.route('/prompts/<prompt_id>/approve', methods=['POST'])
def approve_prompt(prompt_id):
    return jsonify({'message': f'Approve prompt {prompt_id} (stub)'}), 200


@app.route('/prompts/<prompt_id>/generate-image', methods=['POST'])
def generate_image(prompt_id):
    return jsonify({'message': f'Generate image for prompt {prompt_id} (stub)'}), 200


@app.route('/prompts/<prompt_id>/generated-image', methods=['GET'])
def get_generated_image(prompt_id):
    return jsonify({'message': f'Get generated image for prompt {prompt_id} (stub)'}), 200


# FEEDBACK (Question here: feedback on prompts, see 1st route, or generated images [commented-out route]?)
@app.route('/prompts/<prompt_id>/feedback', methods=['POST'])
def give_feedback(prompt_id):
    data = request.get_json()
    rating = data.get('rating')
    comment = data.get('comment', '')

    return jsonify({
        'message': f'Feedback received for prompt {prompt_id} (stub)',
        'rating': rating,
        'comment': comment
    }), 200


# @app.route('/generated_images/<generated_image_id>/feedback', methods=['POST'])
# def give_feedback(generated_image_id):
#     data = request.get_json()
#     rating = data.get('rating')
#     comment = data.get('comment', '')

#     return jsonify({
#         'message': f'Feedback received for prompt {generated_image_id} (stub)',
#         'rating': rating,
#         'comment': comment
#     }), 200


@app.route('/users/me/history', methods=['GET'])
def get_user_history():
    return jsonify({'message': 'User history endpoint (stub)'}), 200


if __name__ == '__main__':
    app.run(debug=True)