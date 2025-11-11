# 🌱 Hướng Dẫn Seed Data cho Intent Analysis

## Tổng Quan

Script `seed_data.py` giúp bạn dễ dàng seed data cho hệ thống Intent Analysis bằng cách call API. Script này không cần kết nối trực tiếp database, chỉ cần services đang chạy trong Docker.

## Yêu Cầu

1. **Docker services đang chạy:**
   ```bash
   docker-compose up -d
   ```

2. **Python 3.7+** và package `requests`:
   ```bash
   pip install requests
   ```

## Cách Sử Dụng

### 1. Seed Tất Cả (Khuyến Nghị Lần Đầu)

Seed embeddings cho labels và intents cho feedbacks mới:

```bash
python seed_data.py
```

### 2. Chỉ Seed Embeddings cho Labels

Nếu bạn chỉ muốn tính embedding cho labels (ví dụ: sau khi thêm labels mới):

```bash
python seed_data.py --labels-only
```

### 3. Chỉ Seed Intents cho Feedbacks

Nếu labels đã có embedding rồi, chỉ cần seed intents cho feedbacks mới:

```bash
python seed_data.py --intents-only
```

### 4. Recompute Tất Cả

Tính lại embedding và intents cho TẤT CẢ data (bao gồm cả data cũ đã có cache):

```bash
python seed_data.py --recompute
```

## Kết Quả Mẫu

```
======================================================================
  🚀 SEED DATA SCRIPT - Intent Analysis System
======================================================================

🕐 Started at: 2024-01-15 10:30:00
🌐 API Base URL: http://localhost:8001/api/v1

----------------------------------------------------------------------
  Checking Services Health
----------------------------------------------------------------------
✅ Label Backend Service: OK

----------------------------------------------------------------------
  Seeding Label Embeddings
----------------------------------------------------------------------
📡 Calling API: POST /admin/seed-label-embeddings
⏳ Processing... (This may take a few minutes)

✅ SUCCESS!
   Total labels: 15
   Processed: 15
   Failed: 0
   Time taken: 12.34 seconds

----------------------------------------------------------------------
  Seeding Feedback Intents
----------------------------------------------------------------------
📡 Calling API: POST /admin/seed-feedback-intents?recompute=False
   Mode: new feedbacks only
⏳ Processing... (This may take a few minutes)

✅ SUCCESS!
   Total feedbacks: 50
   Processed: 50
   Failed: 0
   Time taken: 45.67 seconds

======================================================================
  ✅ ALL OPERATIONS COMPLETED SUCCESSFULLY!
======================================================================

🕐 Finished at: 2024-01-15 10:31:00
```

## API Endpoints Được Sử Dụng

Script này gọi 2 API endpoints:

### 1. Seed Label Embeddings
```http
POST /api/v1/admin/seed-label-embeddings
```

**Chức năng:**
- Lấy tất cả labels từ database
- Gọi embedding service để tính embedding cho mỗi label
- Update embedding vào database

### 2. Seed Feedback Intents
```http
POST /api/v1/admin/seed-feedback-intents?recompute={true|false}
```

**Chức năng:**
- Lấy feedbacks cần xử lý (mới hoặc tất cả tùy theo `recompute`)
- Tính embedding cho mỗi feedback
- Tính top 10 intent triplets
- Cache kết quả vào database

## Troubleshooting

### Lỗi: "Cannot connect to services"

**Nguyên nhân:** Docker services chưa chạy.

**Giải pháp:**
```bash
# Kiểm tra services
docker-compose ps

# Start services nếu chưa chạy
docker-compose up -d

# Chờ services khởi động (khoảng 30s - 1 phút)
sleep 30

# Thử lại
python seed_data.py
```

### Lỗi: "Request timeout"

**Nguyên nhân:** Có quá nhiều labels/feedbacks cần xử lý.

**Giải pháp:**
- Script sẽ tự động timeout sau 10 phút
- Bạn có thể chạy lại script, nó sẽ chỉ xử lý data chưa có (trừ khi dùng `--recompute`)

### Lỗi: "Label embedding failed"

**Nguyên nhân:** Embedding service có vấn đề.

**Giải pháp:**
```bash
# Check logs
docker-compose logs embedding-service

# Restart service
docker-compose restart embedding-service

# Thử lại
python seed_data.py --labels-only
```

### Một số labels/feedbacks bị "Failed"

**Nguyên nhân:** Có thể do:
- Embedding service tạm thời quá tải
- Text rỗng hoặc không hợp lệ
- Lỗi network tạm thời

**Giải pháp:**
- Chạy lại script, nó sẽ xử lý những cái còn thiếu
- Check logs backend để xem chi tiết:
  ```bash
  docker-compose logs label-backend
  ```

## Khi Nào Cần Chạy Script?

### Bắt Buộc:
1. **Lần đầu khởi động hệ thống** - Cần seed embeddings cho labels hiện có
2. **Sau khi thêm labels mới** - Chạy `--labels-only`

### Tùy Chọn:
1. **Định kỳ** - Seed intents cho feedbacks mới (có thể setup cron job)
2. **Sau khi update embedding model** - Chạy với `--recompute` để tính lại tất cả

## Tự Động Hóa (Cron Job)

Nếu muốn tự động seed intents cho feedbacks mới mỗi ngày:

```bash
# Mở crontab
crontab -e

# Thêm dòng này (chạy lúc 2h sáng mỗi ngày)
0 2 * * * cd /path/to/project && python seed_data.py --intents-only >> /var/log/seed_data.log 2>&1
```

## Lưu Ý

1. **Thời gian xử lý:** Tùy thuộc số lượng labels/feedbacks (khoảng 1-2s per item)
2. **Idempotent:** Script có thể chạy nhiều lần an toàn (không duplicate data)
3. **Incremental:** Mặc định chỉ xử lý data mới (trừ khi dùng `--recompute`)
4. **Network:** Script cần kết nối đến `localhost:8001` (label-backend)

## Kiểm Tra Kết Quả

Sau khi seed xong, kiểm tra trong database:

```sql
-- Số labels có embedding
SELECT COUNT(*) FROM labels WHERE embedding IS NOT NULL;

-- Số feedbacks có intent analysis
SELECT COUNT(DISTINCT feedback_id) FROM feedback_intents;

-- Top 10 intent triplets phổ biến nhất
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

## Support

Nếu gặp vấn đề, check logs:

```bash
# Backend logs
docker-compose logs -f label-backend

# Embedding service logs
docker-compose logs -f embedding-service

# All services
docker-compose logs -f
```


