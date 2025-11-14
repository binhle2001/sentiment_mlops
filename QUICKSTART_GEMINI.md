# 🚀 Quick Start - Gemini Integration

## Bước 1: Lấy Gemini API Key

Truy cập: https://makersuite.google.com/app/apikey

Click "Create API Key" và copy key.

## Bước 2: Add vào `.env`

```bash
# Thêm vào cuối file .env
GEMINI_API_KEY=AIzaSy...your_key_here...
```

## Bước 3: Chạy Migration

```bash
pip install psycopg2-binary python-dotenv google-generativeai
python migrate_v2.py
```

## Bước 4: Rebuild Services

```bash
docker-compose up -d --build
```

## Bước 5: Seed Embeddings (Nếu Chưa)

```bash
python seed_data.py --labels-only
```

## Bước 6: Test!

### Via UI:
1. Mở http://localhost:2345
2. Submit feedback: "Chuyển tiền bị lỗi"
3. Xem kết quả có cả intent (Level 1 → 2 → 3)

### Via API:
```bash
curl -X POST http://localhost:3456/api/v1/feedbacks \
  -H "Content-Type: application/json" \
  -d '{
    "feedback_text": "Chuyển tiền bị lỗi",
    "feedback_source": "app"
  }'
```

## Verify

```sql
-- Check feedbacks với intent
SELECT 
    fs.feedback_text,
    l1.name as level1,
    l2.name as level2,
    l3.name as level3
FROM feedback_sentiments fs
JOIN labels l1 ON fs.level1_id = l1.id
JOIN labels l2 ON fs.level2_id = l2.id
JOIN labels l3 ON fs.level3_id = l3.id
ORDER BY fs.created_at DESC
LIMIT 10;
```

## Troubleshooting

**Intent luôn NULL?**
→ Check: `docker-compose logs label-backend | grep "Gemini"`

**Labels chưa có embedding?**
→ Run: `python seed_data.py --labels-only`

**API key sai?**
→ Check: `cat .env | grep GEMINI_API_KEY`

Done! 🎉

