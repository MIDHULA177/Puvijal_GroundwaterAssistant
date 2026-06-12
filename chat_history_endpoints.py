from flask import Blueprint, request, jsonify
from datetime import datetime
from bson import ObjectId
import jwt
import os

chat_history_bp = Blueprint('chat_history', __name__)

# These are injected when the blueprint is registered via init_app()
_db = None
_secret = None

def init_app(db, secret_key):
    global _db, _secret
    _db = db
    _secret = secret_key

def _get_email():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    try:
        payload = jwt.decode(token, _secret, algorithms=['HS256'])
        return payload.get('email')
    except:
        return None

@chat_history_bp.route('/chat/history', methods=['POST'])
def save_chat_history():
    email = _get_email()
    if not email:
        return jsonify({'error': 'Invalid token'}), 401
    chats = request.json.get('chats', {})
    _db.users.update_one(
        {'email': email},
        {'$set': {'chat_history': chats, 'updated_at': datetime.utcnow()}},
        upsert=True
    )
    return jsonify({'message': 'Chat history saved'}), 200

@chat_history_bp.route('/chat/history', methods=['GET'])
def get_chat_history():
    email = _get_email()
    if not email:
        return jsonify({'error': 'Invalid token'}), 401
    user = _db.users.find_one({'email': email}, {'chat_history': 1})
    return jsonify({'chats': user.get('chat_history', {}) if user else {}})
