# 🔄 Database Migration Guide

## Tổng Quan

File `migrate.py` là Python script để chạy database migration cho tính năng Intent Analysis. Script này sẽ:
- Thêm cột `embedding` vào bảng `labels`
- Tạo bảng `feedback_intents` để cache kết quả phân tích
- Tạo các indexes cần thiết

Ngoài ra, để chuyển toàn bộ `labels.id` từ UUID sang số nguyên (phục vụ đồng bộ với hệ thống khác), sử dụng script mới `migrate_ids_to_int.py`. Script này sẽ:
- Sinh ID nguyên tăng dần dựa trên thứ tự hiện tại của bảng `labels`
- Cập nhật toàn bộ khóa ngoại liên quan (`feedback_sentiments`, `feedback_intents`) sang INTEGER
- Tái tạo constraint/index tương ứng

👉 **Chạy script này ngay sau khi pull phiên bản mới và trước khi khởi động dịch vụ.**

```bash
python migrate_ids_to_int.py
```

## Yêu Cầu

1. **Docker services đang chạy:**
   ```bash
   docker-compose up -d
   ```

2. **Python packages:**
   ```bash
   pip install psycopg2-binary python-dotenv
   ```

3. **File `.env` với đầy đủ config database**

## Cách Sử Dụng

### 1. Chạy Migration

```bash
python migrate.py
```

### 2. Kết Quả Mong Đợi

```
🚀 Starting database migration...
   Target database: label_db@localhost:5432

✅ Connected to database successfully

🔍 Checking prerequisites...
   ✅ Table 'labels' exists
   ✅ Table 'feedback_sentiments' exists

======================================================================
  DATABASE MIGRATION - Intent Analysis Feature
======================================================================

📝 Step 1: Adding embedding column to labels table...
   ✅ Column 'embedding' added to labels table

📝 Step 2: Creating index on embedding column...
   ✅ Index 'idx_labels_embedding' created

📝 Step 3: Creating feedback_intents table...
   ✅ Table 'feedback_intents' created

📝 Step 4: Creating indexes on feedback_intents...
   ✅ Index 'idx_feedback_intents_feedback_id' created
   ✅ Index 'idx_feedback_intents_level1_id' created
   ✅ Index 'idx_feedback_intents_level2_id' created
   ✅ Index 'idx_feedback_intents_level3_id' created
   ✅ Index 'idx_feedback_intents_similarity' created
   ✅ Index 'idx_feedback_intents_created_at' created

📝 Step 5: Adding documentation comments...
   ✅ Comments added

======================================================================
  ✅ MIGRATION COMPLETED SUCCESSFULLY!
======================================================================

📊 Verification:
   ✅ labels.embedding column exists
   ✅ feedback_intents table exists
   📊 Labels with embeddings: 0
   📊 Feedbacks with cached intents: 0

🎯 Next steps:
   1. Run: python seed_data.py --labels-only
   2. Run: python seed_data.py --intents-only

✅ Database connection closed
```

## Chi Tiết Migration

### Thay Đổi Schema

**1. Bảng `labels`:**
```sql
-- Thêm cột mới
ALTER TABLE labels ADD COLUMN embedding REAL[];

-- Index
CREATE INDEX idx_labels_embedding ON labels USING GIN(embedding);
```

**2. Bảng `feedback_intents` (mới):**
```sql
CREATE TABLE feedback_intents (
    id UUID PRIMARY KEY,
    feedback_id UUID NOT NULL,
    level1_id INTEGER NOT NULL,
    level2_id INTEGER NOT NULL,
    level3_id INTEGER NOT NULL,
    avg_cosine_similarity REAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE,
    
    -- Foreign keys
    FOREIGN KEY (feedback_id) REFERENCES feedback_sentiments(id),
    FOREIGN KEY (level1_id) REFERENCES labels(id),
    FOREIGN KEY (level2_id) REFERENCES labels(id),
    FOREIGN KEY (level3_id) REFERENCES labels(id),
    
    -- Unique constraint
    UNIQUE (feedback_id, level1_id, level2_id, level3_id)
);
```

**3. Indexes:**
- `idx_feedback_intents_feedback_id`
- `idx_feedback_intents_level1_id`
- `idx_feedback_intents_level2_id`
- `idx_feedback_intents_level3_id`
- `idx_feedback_intents_similarity`
- `idx_feedback_intents_created_at`

## Troubleshooting

### Lỗi: "Failed to connect to database"

**Nguyên nhân:** Services chưa chạy hoặc config sai.

**Giải pháp:**
```bash
# Kiểm tra services
docker-compose ps

# Kiểm tra .env file
cat .env | grep POSTGRES

# Start services nếu chưa chạy
docker-compose up -d
```

### Lỗi: "Table 'labels' does not exist"

**Nguyên nhân:** Database chưa được init với schema ban đầu.

**Giải pháp:**
```bash
# Chạy init migration trước
docker-compose exec postgres psql -U labeluser -d label_db -f /docker-entrypoint-initdb.d/01-init.sql
```

### Lỗi: "Table 'feedback_sentiments' does not exist"

**Nguyên nhân:** Bảng feedback_sentiments chưa được tạo.

**Giải pháp:** Tạo bảng này trước:
```sql
CREATE TABLE feedback_sentiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feedback_text TEXT NOT NULL,
    sentiment_label VARCHAR(50) NOT NULL,
    confidence_score REAL NOT NULL,
    feedback_source VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### Migration Đã Chạy Rồi (Idempotent)

Migration sử dụng `IF NOT EXISTS`, nên có thể chạy nhiều lần an toàn:
- Nếu cột/bảng đã tồn tại → Skip
- Nếu chưa tồn tại → Tạo mới

### Rollback Migration

Nếu cần rollback:

```bash
docker-compose exec postgres psql -U labeluser -d label_db << 'EOF'
-- Drop feedback_intents table
DROP TABLE IF EXISTS feedback_intents CASCADE;

-- Drop embedding column
ALTER TABLE labels DROP COLUMN IF EXISTS embedding;

-- Drop index
DROP INDEX IF EXISTS idx_labels_embedding;
EOF
```

## Verify Migration

Sau khi chạy migration, verify:

```bash
# Kiểm tra structure
docker-compose exec postgres psql -U labeluser -d label_db << 'EOF'
-- Check labels table has embedding column
\d labels

-- Check feedback_intents table exists
\d feedback_intents

-- Count labels with embeddings
SELECT COUNT(*) FROM labels WHERE embedding IS NOT NULL;

-- Count cached intents
SELECT COUNT(DISTINCT feedback_id) FROM feedback_intents;
EOF
```

## Next Steps

Sau khi migration thành công:

1. **Seed embeddings cho labels:**
   ```bash
   python seed_data.py --labels-only
   ```

2. **Seed intents cho feedbacks:**
   ```bash
   python seed_data.py --intents-only
   ```

3. **Hoặc seed tất cả:**
   ```bash
   python seed_data.py
   ```

## Environment Variables

Script sử dụng các biến môi trường sau từ file `.env`:

```bash
POSTGRES_HOST=localhost      # hoặc postgres nếu chạy trong Docker
POSTGRES_PORT=5432
POSTGRES_DB=label_db
POSTGRES_USER=labeluser
POSTGRES_PASSWORD=labelpass123
```

## Support

Nếu gặp vấn đề:

1. Kiểm tra logs:
   ```bash
   docker-compose logs postgres
   ```

2. Test connection:
   ```bash
   docker-compose exec postgres psql -U labeluser -d label_db -c "SELECT version();"
   ```

3. Check tables:
   ```bash
   docker-compose exec postgres psql -U labeluser -d label_db -c "\dt"
   ```

