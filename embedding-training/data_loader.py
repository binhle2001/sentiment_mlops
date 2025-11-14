import random
from typing import List, Dict, Optional
from database import get_db, execute_query

def get_training_data() -> List[Dict[str, Optional[str]]]:
    """Fetch and structure training data from the database."""
    with get_db() as conn:
        # Lấy tất cả các feedback đã được gán nhãn
        feedbacks = execute_query(conn, """
            SELECT fs.feedback_text, l.name as intent
            FROM feedback_sentiments fs
            JOIN labels l ON fs.level3_id = l.id
            WHERE fs.level3_id IS NOT NULL AND fs.is_model_confirmed = TRUE;
        """, fetch="all")
        
        # Lấy tất cả các intent
        all_intents = execute_query(conn, "SELECT name FROM labels;", fetch="all")
        all_intent_names = [intent['name'] for intent in all_intents]
        
        training_examples = []
        for feedback in feedbacks:
            query = feedback['feedback_text']
            positive = feedback['intent']
            
            # Chọn một negative ngẫu nhiên từ các intent khác
            possible_negatives = [name for name in all_intent_names if name != positive]
            if possible_negatives:
                negative = random.choice(possible_negatives)
            else:
                negative = None # Không có negative nào khác
            
            training_examples.append({
                "query": query,
                "pos": positive,
                "neg": negative
            })
            
    return training_examples