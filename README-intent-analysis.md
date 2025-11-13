# Hướng Dẫn Sử Dụng Tính Năng Intent Analysis

## Tổng Quan

Tính năng Intent Analysis cho phép phân tích ý định (intent) của feedback khách hàng bằng cách sử dụng embedding service và tính toán độ tương đồng cosine với các label trong hệ thống.

Khi người dùng submit một feedback, hệ thống sẽ:
1. Phân tích sentiment (tích cực/tiêu cực/trung tính)
2. Tính toán embedding cho feedback text
3. Sử dụng thuật toán hierarchical top-down:
   - Tìm top 5 level1 có similarity cao nhất
   - Với mỗi level1, tìm top 4 level2 con → ~20 level2
   - Với mỗi level2, tìm top 2-3 level3 con → ~50 triplets
4. Trả về top 50 intent triplets có độ tương đồng cao nhất

## Cài Đặt & Triển Khai

### 1. Cấu Hình Environment Variables

Thêm các biến môi trường sau vào file `.env`:

```bash
# Embedding Service
EMBEDDING_EXTERNAL_PORT=8000
EMBEDDING_PORT=8000
EMBEDDING_SERVICE_URL=http://embedding-service:8000/api/v1
```

### 2. Khởi Động Services

```bash
# Build và start tất cả services
docker-compose up -d --build

# Kiểm tra services đang chạy
docker-compose ps

# Xem logs
docker-compose logs -f embedding-service
docker-compose logs -f label-backend
```

### 3. Chạy Database Migration

Database migration sẽ tự động chạy khi khởi động PostgreSQL container. File migration:
- `db/init/02-add-embedding.sql`

Nếu cần chạy lại migration thủ công:

```bash
docker-compose exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -f /docker-entrypoint-initdb.d/02-add-embedding.sql
```

## Tính Embedding Cho Labels

### Bước 1: Cài Đặt Dependencies

```bash
cd scripts
pip install -r requirements.txt
```

### Bước 2: Cấu Hình Environment

Tạo file `.env` trong thư mục `scripts` hoặc copy từ root:

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=label_db
POSTGRES_USER=labeluser
POSTGRES_PASSWORD=labelpass123

# Service URLs
EMBEDDING_SERVICE_URL=http://localhost:8000/api/v1
LABEL_BACKEND_URL=http://localhost:8001/api/v1
```

### Bước 3: Chạy Script Tính Embedding

```bash
# Tính embedding cho tất cả labels
python scripts/compute_label_embeddings.py
```

Script này sẽ:
- Kết nối đến database
- Lấy tất cả labels
- Gọi embedding service để tính embedding cho mỗi label
- Lưu embedding vào database

**Lưu ý:** Cần chạy script này trước khi phân tích intent cho feedbacks.

## Tính Intent Cho Feedbacks

### Chạy Script Tính Intent

```bash
# Tính intent cho feedbacks chưa có cache
python scripts/compute_feedback_intents.py

# Tính lại intent cho tất cả feedbacks (bao gồm cả những cái đã có cache)
python scripts/compute_feedback_intents.py --recompute
```

Script này sẽ:
- Lấy tất cả feedbacks cần tính intent
- Gọi API backend để tính và cache kết quả
- Hiển thị progress bar và thống kê

## Sử Dụng Trên Giao Diện

### 1. Submit Feedback Mới

1. Truy cập trang "Phân tích Sentiment Feedback"
2. Nhập nội dung feedback
3. Chọn nguồn feedback (Web, App, Map, v.v.)
4. Click "Phân tích Sentiment"

Kết quả sẽ hiển thị:
- **Sentiment**: Tích cực/Tiêu cực/Trung tính
- **Độ tin cậy**: Confidence score
- **Nguồn**: Nguồn feedback
- **Top 50 Intent Triplets**: Danh sách các intent path với độ tương đồng cao nhất (theo thuật toán hierarchical)

### 2. Hiểu Kết Quả Intent Analysis

Intent triplet được hiển thị dạng:

```
Level 1 → Level 2 → Level 3     Độ tương đồng: XX.XX%
```

Ví dụ:
```
Dịch vụ → Giao dịch → Chuyển tiền     Độ tương đồng: 85.23%
```

Màu sắc độ tương đồng:
- 🟢 Xanh (≥ 70%): Độ tương đồng cao
- 🟠 Cam (≥ 50%): Độ tương đồng trung bình
- ⚪ Xám (< 50%): Độ tương đồng thấp

## API Endpoints

### 1. Phân Tích Intent Cho Feedback

```http
POST /api/v1/feedbacks/{feedback_id}/intents
```

**Response:**
```json
{
  "feedback_id": "uuid",
  "intents": [
    {
      "level1": {
        "id": "uuid",
        "name": "Dịch vụ",
        "level": 1,
        ...
      },
      "level2": {
        "id": "uuid",
        "name": "Giao dịch",
        "level": 2,
        ...
      },
      "level3": {
        "id": "uuid",
        "name": "Chuyển tiền",
        "level": 3,
        ...
      },
      "avg_cosine_similarity": 0.8523
    },
    ...
  ],
  "total_intents": 10
}
```

### 2. Lấy Intent Đã Cache

```http
GET /api/v1/feedbacks/{feedback_id}/intents
```

Trả về kết quả intent đã được cache trước đó.

## Thuật Toán Tính Intent

### Hierarchical Top-Down Approach

Thay vì tính toán tất cả các triplets có thể (rất tốn thời gian), hệ thống sử dụng thuật toán **hierarchical top-down**:

**Bước 1: Tìm Top 5 Level1**
1. Tính embedding cho feedback text: `E_feedback`
2. Tính cosine similarity với TẤT CẢ level1 labels
3. Sắp xếp giảm dần và lấy **top 5 level1**

**Bước 2: Tìm Top ~20 Level2**
4. Với mỗi level1 trong top 5:
   - Lấy các level2 con (children) của level1 đó
   - Tính cosine similarity với E_feedback
   - Lấy **top 4 level2** cho mỗi level1
5. Kết quả: ~20 level2 (5 × 4 = 20)

**Bước 3: Tìm Top ~50 Level3 (Triplets)**
6. Với mỗi level2 trong top 20:
   - Lấy các level3 con (children) của level2 đó
   - Tính cosine similarity với E_feedback
   - Lấy **top 2-3 level3** cho mỗi level2
7. Kết quả: ~50 triplets (20 × 2.5 ≈ 50)

**Bước 4: Tính Average Similarity và Sắp Xếp**
8. Với mỗi triplet (level1, level2, level3):
   - Tính: `avg_sim = (sim1 + sim2 + sim3) / 3`
9. Sắp xếp theo `avg_sim` giảm dần
10. Trả về **top 50 triplets**

### Ưu Điểm

- ⚡ **Nhanh hơn**: Chỉ tính similarity cho subset nhỏ thay vì tất cả combinations
- 🎯 **Chính xác hơn**: Focus vào những nhánh có similarity cao
- 📊 **Diverse**: Đảm bảo có triplets từ nhiều nhánh level1 khác nhau

### Cosine Similarity

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Giá trị từ -1 đến 1, trong đó:
- 1: Hoàn toàn giống nhau
- 0: Không liên quan
- -1: Hoàn toàn trái ngược

## Troubleshooting

### Lỗi: "Embedding service is unavailable"

**Giải pháp:**
```bash
# Kiểm tra embedding service
docker-compose logs embedding-service

# Restart service
docker-compose restart embedding-service
```

### Lỗi: "No intents found"

**Nguyên nhân:** Labels chưa có embedding.

**Giải pháp:**
```bash
python scripts/compute_label_embeddings.py
```

### Lỗi: Database connection failed

**Giải pháp:**
```bash
# Kiểm tra PostgreSQL
docker-compose ps postgres

# Kiểm tra logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

### Performance Issues

Nếu việc tính intent chậm:

1. **Tăng số workers cho embedding service:**
   - Edit `embedding/config.py`: `workers: int = 2`

2. **Batch processing cho nhiều feedbacks:**
   - Sử dụng script với batch size nhỏ hơn

3. **Cache kết quả:**
   - Intents đã tính sẽ được cache trong bảng `feedback_intents`
   - Sử dụng GET endpoint để lấy cached results

## Maintenance

### Backup Database

```bash
docker-compose exec postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB > backup.sql
```

### Clear Intent Cache

```sql
-- Xóa tất cả cached intents
TRUNCATE TABLE feedback_intents;

-- Xóa cached intents cho một feedback
DELETE FROM feedback_intents WHERE feedback_id = 'uuid';
```

### Update Label Embeddings

Khi thêm hoặc sửa labels, cần chạy lại:

```bash
python scripts/compute_label_embeddings.py
```

## Monitoring

### Kiểm Tra Health

```bash
# Embedding Service
curl http://localhost:8000/api/v1/health

# Label Backend
curl http://localhost:8001/api/v1/health
```

### Database Statistics

```sql
-- Số labels có embedding
SELECT COUNT(*) FROM labels WHERE embedding IS NOT NULL;

-- Số feedbacks có intent analysis
SELECT COUNT(DISTINCT feedback_id) FROM feedback_intents;

-- Top intents được sử dụng
SELECT 
    l1.name as level1,
    l2.name as level2,
    l3.name as level3,
    COUNT(*) as count
FROM feedback_intents fi
JOIN labels l1 ON fi.level1_id = l1.id
JOIN labels l2 ON fi.level2_id = l2.id
JOIN labels l3 ON fi.level3_id = l3.id
GROUP BY l1.name, l2.name, l3.name
ORDER BY count DESC
LIMIT 10;
```

## Trigger Huấn Luyện Tự Động (Intent & Sentiment)

- Bật/tắt bằng biến môi trường `ENABLE_TRAINING_TRIGGER` (mặc định bật).
- Sau khi người dùng xác nhận (`is_model_confirmed=True`) hoặc chỉnh sửa intent/sentiment, backend đếm số lần `confirmed`/`relabel`.
- Điều kiện kích hoạt (mặc định): `confirmed > 200` và `relabel > 30`, đồng thời không có sự kiện mới trong 60 giây (`TRAINING_IDLE_SECONDS`).
- Mỗi model (intent hoặc sentiment) sẽ lần lượt chiếm biến trigger; model còn lại phải chờ tới lượt.
- URL mặc định:
  - Sentiment training: `http://sentiment-training-service:8010/api/v1/train`
  - Intent (embedding) training: `http://embedding-training-service:8001/api/train`
- API giám sát: `GET /api/v1/training/status` trả về `current_trigger` và bộ đếm hiện tại.

### Gợi ý kiểm thử thủ công

1. Chạy `docker-compose up label-backend sentiment-training-service embedding-training-service`.
2. Tạo feedback, chỉnh sửa intent vài lần → gọi `/api/v1/training/status` để xem bộ đếm tăng.
3. Xác nhận feedback (`POST /api/v1/feedbacks/{id}/confirm`) nhiều lần cho tới khi vượt ngưỡng, chờ 60 giây → backend sẽ gọi service train tương ứng (xem log).
4. Thử thay đổi `sentiment_label` để quan sát trigger huấn luyện sentiment.
5. Khi dừng service (`docker-compose down`), kiểm tra log đảm bảo TrainingManager shutdown gọn.

## Tài Liệu Tham Khảo

- **BGE-M3 Model**: Embedding model được sử dụng (1024 dimensions)
- **Cosine Similarity**: Phương pháp đo độ tương đồng giữa các vectors
- **PostgreSQL ARRAY Type**: Lưu trữ embedding vectors trong database

## Liên Hệ & Hỗ Trợ

Nếu gặp vấn đề hoặc cần hỗ trợ, vui lòng tạo issue hoặc liên hệ team phát triển.


