"""
Gemini AI service for intent classification.
Uses Gemini 2.5-flash-lite to select the best intent triplet.
"""
import os
import json
import logging
import google.generativeai as genai
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini service."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use gemini-2.0-flash-exp model (latest available)
        # Note: gemini-2.5-flash-lite might not be available yet
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        logger.info("Gemini service initialized successfully")
    
    def select_best_intent(
        self, 
        feedback_text: str, 
        intent_candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Use Gemini to select the best intent triplet from candidates.
        
        Args:
            feedback_text: The customer feedback text
            intent_candidates: List of top 10 intent triplets with similarity scores
            
        Returns:
            Selected intent triplet dict with level1, level2, level3
        """
        try:
            # Build prompt
            prompt = self._build_prompt(feedback_text, intent_candidates)
            
            # Call Gemini API
            logger.info(f"Calling Gemini API for feedback: {feedback_text[:50]}...")
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,  # Low temperature for more deterministic output
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 1024,
                }
            )
            
            # Parse response
            result = self._parse_response(response.text, intent_candidates)
            
            if result:
                logger.info(f"Gemini selected: {result['level1']['name']} → "
                          f"{result['level2']['name']} → {result['level3']['name']}")
            else:
                logger.warning("Gemini failed to select a valid intent")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}", exc_info=True)
            return None
    
    def _build_prompt(
        self, 
        feedback_text: str, 
        candidates: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for Gemini."""
        
        # Format candidates
        candidates_text = ""
        for i, candidate in enumerate(candidates, 1):
            l1 = candidate['level1']['name']
            l2 = candidate['level2']['name']
            l3 = candidate['level3']['name']
            sim = candidate['avg_cosine_similarity']
            
            candidates_text += f"{i}. {l1} → {l2} → {l3} (similarity: {sim:.4f})\n"
        
        prompt = f"""Bạn là một chuyên gia phân loại phản hồi khách hàng cho ngân hàng.

NHIỆM VỤ: Phân tích phản hồi của khách hàng và chọn nhãn phân loại phù hợp nhất từ danh sách ứng viên.

PHẢN HỒI KHÁCH HÀNG:
"{feedback_text}"

DANH SÁCH 10 NHÃN ỨNG VIÊN (đã được sắp xếp theo độ tương đồng):
{candidates_text}

HƯỚNG DẪN:
1. Đọc kỹ nội dung phản hồi của khách hàng
2. Xem xét các nhãn ứng viên (mỗi nhãn có 3 cấp độ: Cấp 1 → Cấp 2 → Cấp 3)
3. Chọn nhãn phù hợp nhất dựa trên:
   - Ý nghĩa ngữ nghĩa của phản hồi
   - Độ tương đồng (similarity score)
   - Mối quan hệ giữa 3 cấp độ nhãn
4. Trả về CHÍNH XÁC số thứ tự của nhãn được chọn (1-10)

YÊU CẦU ĐẦU RA:
Trả về ĐÚNG JSON format sau (không thêm text nào khác):
{{
    "selected_index": <số từ 1-10>,
    "reasoning": "<giải thích ngắn gọn tại sao chọn nhãn này>"
}}

Ví dụ:
{{
    "selected_index": 1,
    "reasoning": "Phản hồi đề cập trực tiếp đến vấn đề chuyển tiền trong giao dịch ngân hàng"
}}"""
        
        return prompt
    
    def _parse_response(
        self, 
        response_text: str, 
        candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Parse Gemini response and return selected intent."""
        try:
            # Clean response text
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            data = json.loads(response_text)
            
            # Validate
            if 'selected_index' not in data:
                logger.error("Response missing 'selected_index' field")
                return None
            
            selected_idx = int(data['selected_index'])
            
            # Validate index
            if selected_idx < 1 or selected_idx > len(candidates):
                logger.error(f"Invalid selected_index: {selected_idx}")
                return None
            
            # Get selected candidate (index is 1-based)
            selected = candidates[selected_idx - 1]
            
            # Add reasoning if available
            if 'reasoning' in data:
                selected['gemini_reasoning'] = data['reasoning']
            
            return selected
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Response text: {response_text}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Error processing Gemini response: {e}")
            return None


# Singleton instance
_gemini_service: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Get or create Gemini service singleton."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service

