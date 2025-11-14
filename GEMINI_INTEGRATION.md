# 🤖 Gemini AI Integration Guide

## Tổng Quan

Hệ thống sử dụng **Gemini 2.0-flash-exp** để tự động phân loại intent của feedback khách hàng. Gemini AI sẽ chọn nhãn phù hợp nhất từ top 10 candidates.

## Flow Hoàn Chỉnh

```
┌────────────────────────────────────────────────────────────┐
│  1. User submit feedback: "Chuyển tiền bị lỗi"            │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  2. Sentiment Analysis: NEGATIVE (0.95 confidence)         │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  3. Embedding Service: [0.123, 0.456, ...]  (1024 dims)   │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  4. Intent Candidates (Hierarchical Algorithm):            │
│     • Top 5 Level1 (by similarity)                         │
│     • Top 15 Level2 from 5 L1 (cross all)                  │
│     • Top 50 Level3 from 15 L2 (cross all)                 │
│     • Rerank by avg similarity → Top 10                    │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  5. Gemini AI Selection:                                   │
│                                                             │
│     Input: Feedback text + 10 intent triplets             │
│     Output: Selected triplet (Level 1, 2, 3)              │
│                                                             │
│     Ví dụ: Dịch vụ → Giao dịch → Chuyển tiền            │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  6. Save to Database:                                       │
│     • feedback_sentiments table                             │
│     • level1_id, level2_id, level3_id (foreign keys)       │
└────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Lấy Gemini API Key

1. Truy cập: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy API key

### 2. Thêm vào `.env`

```bash
# Gemini AI Configuration
GEMINI_API_KEY=AIzaSy...your_api_key_here...
```

### 3. Chạy Migration V2

```bash
# Cài đặt dependencies
pip install psycopg2-binary python-dotenv

# Chạy migration để thêm 3 cột vào feedback_sentiments
python migrate_v2.py
```

### 4. Rebuild Services

```bash
# Rebuild label-backend với Gemini integration
docker-compose up -d --build label-backend

# Check logs
docker-compose logs -f label-backend
```

## Testing

### Test Qua UI

1. Truy cập: http://localhost:2345
2. Vào trang "Phân tích Sentiment Feedback"
3. Nhập feedback: "Chuyển tiền bị lỗi"
4. Submit

Kết quả sẽ hiển thị:
- ✅ Sentiment: NEGATIVE
- ✅ Độ tin cậy: 95%
- ✅ **Intent: Dịch vụ → Giao dịch → Chuyển tiền** ← MỚI

### Test Qua API

```bash
curl -X POST http://localhost:3456/api/v1/feedbacks \
  -H "Content-Type: application/json" \
  -d '{
    "feedback_text": "Chuyển tiền bị lỗi",
    "feedback_source": "app"
  }'
```

Response:

```json
{
  "id": "uuid",
  "feedback_text": "Chuyển tiền bị lỗi",
  "sentiment_label": "NEGATIVE",
  "confidence_score": 0.95,
  "feedback_source": "app",
  "created_at": "2024-11-11T...",
  "level1_id": "uuid-level1",
  "level2_id": "uuid-level2",
  "level3_id": "uuid-level3",
  "level1_name": "Dịch vụ",
  "level2_name": "Giao dịch",
  "level3_name": "Chuyển tiền"
}
```

## Gemini Prompt

Hệ thống sử dụng prompt sau để guide Gemini:

```
Bạn là một chuyên gia phân loại phản hồi khách hàng cho ngân hàng.

NHIỆM VỤ: Phân tích phản hồi của khách hàng và chọn nhãn phù hợp nhất.

PHẢN HỒI KHÁCH HÀNG:
"{feedback_text}"

DANH SÁCH 10 NHÃN ỨNG VIÊN:
1. Dịch vụ → Giao dịch → Chuyển tiền (similarity: 0.8523)
2. Dịch vụ → Giao dịch → Rút tiền (similarity: 0.7234)
...

YÊU CẦU: Trả về JSON format
{
    "selected_index": 1,
    "reasoning": "Phản hồi đề cập trực tiếp đến vấn đề chuyển tiền"
}
```

## Database Schema

```sql
ALTER TABLE feedback_sentiments ADD COLUMN level1_id UUID;
ALTER TABLE feedback_sentiments ADD COLUMN level2_id UUID;
ALTER TABLE feedback_sentiments ADD COLUMN level3_id UUID;

ALTER TABLE feedback_sentiments 
  ADD CONSTRAINT fk_level1 FOREIGN KEY (level1_id) REFERENCES labels(id);
ALTER TABLE feedback_sentiments 
  ADD CONSTRAINT fk_level2 FOREIGN KEY (level2_id) REFERENCES labels(id);
ALTER TABLE feedback_sentiments 
  ADD CONSTRAINT fk_level3 FOREIGN KEY (level3_id) REFERENCES labels(id);
```

## Error Handling

Hệ thống có graceful fallback:

1. **Embedding service fail** → Lưu feedback không có intent
2. **No intent candidates** → Lưu feedback không có intent
3. **Gemini API fail** → Lưu feedback không có intent
4. **Invalid Gemini response** → Lưu feedback không có intent

Trong mọi trường hợp, feedback vẫn được lưu với sentiment analysis.

## Monitoring

### Check Gemini Usage

```sql
-- Số feedbacks có intent (thành công)
SELECT COUNT(*) FROM feedback_sentiments WHERE level1_id IS NOT NULL;

-- Số feedbacks không có intent (fallback)
SELECT COUNT(*) FROM feedback_sentiments WHERE level1_id IS NULL;

-- Top intents được chọn
SELECT 
    l1.name as level1,
    l2.name as level2,
    l3.name as level3,
    COUNT(*) as count
FROM feedback_sentiments fs
JOIN labels l1 ON fs.level1_id = l1.id
JOIN labels l2 ON fs.level2_id = l2.id
JOIN labels l3 ON fs.level3_id = l3.id
GROUP BY l1.name, l2.name, l3.name
ORDER BY count DESC
LIMIT 10;
```

### Check Logs

```bash
# Xem Gemini API calls
docker-compose logs label-backend | grep "Gemini"

# Xem errors
docker-compose logs label-backend | grep "ERROR"
```

## Cost Optimization

Gemini 2.0-flash-exp là model rất rẻ:
- **Input**: $0.075 per 1M tokens
- **Output**: $0.30 per 1M tokens

Mỗi feedback:
- ~200 tokens input (prompt + candidates)
- ~50 tokens output (JSON response)
- **Cost**: ~$0.000025 per feedback (~25 μ$/feedback)

→ 1 triệu feedbacks = ~$25 💰

## Troubleshooting

### Lỗi: "GEMINI_API_KEY not found"

**Giải pháp:**
```bash
# Check .env file
cat .env | grep GEMINI_API_KEY

# Nếu chưa có, thêm vào
echo "GEMINI_API_KEY=your_key_here" >> .env

# Rebuild
docker-compose up -d --build label-backend
```

### Lỗi: "Gemini service error"

**Nguyên nhân:** API key sai hoặc quota exceeded

**Giải pháp:**
```bash
# Test API key
curl -X POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=YOUR_API_KEY \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"Hello"}]}]}'
```

### Intent luôn NULL

**Nguyên nhân:** Labels chưa có embedding

**Giải pháp:**
```bash
python seed_data.py --labels-only
```

## Best Practices

1. **Monitor Gemini failures**: Log và alert nếu > 10% fails
2. **Validate embeddings**: Đảm bảo tất cả labels có embedding
3. **Review selected intents**: Định kỳ check xem Gemini chọn có đúng không
4. **Adjust candidates**: Có thể thay đổi top 10 → top 5 để giảm cost
5. **Cache results**: Feedback giống nhau có thể reuse intent

## Next Steps

- [ ] Add Gemini reasoning vào database để audit
- [ ] Build dashboard để visualize intent distribution
- [ ] A/B test different prompts
- [ ] Fine-tune model dựa trên feedback

