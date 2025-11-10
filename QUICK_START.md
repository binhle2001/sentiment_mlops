# 🚀 Quick Start - Embedding Service Setup

## Lỗi Gặp Phải

```
WARN[0000] The "EMBEDDING_PORT" variable is not set. Defaulting to a blank string.
WARN[0000] The "EMBEDDING_EXTERNAL_PORT" variable is not set. Defaulting to a blank string.
```

## Giải Pháp

### Bước 1: Thêm biến môi trường vào file `.env`

Mở file `.env` trong thư mục root của project và **thêm** các dòng sau:

```bash
# Embedding Service Configuration
# Dùng port 8003 vì 8002 đã dùng cho sentiment service
EMBEDDING_EXTERNAL_PORT=8003
```

### Bước 2: Verify file `.env` đầy đủ

File `.env` của bạn nên có ít nhất các biến sau:

```bash
# PostgreSQL Database Configuration
POSTGRES_USER=labeluser
POSTGRES_PASSWORD=labelpass123
POSTGRES_DB=label_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Sentiment Service Configuration
SENTIMENT_EXTERNAL_PORT=8002
SENTIMENT_PORT=8005

# Embedding Service Configuration (MỚI)
# Port 8003 vì 8002 đã dùng cho sentiment
EMBEDDING_EXTERNAL_PORT=8003

# Label Backend Service Configuration
LABEL_BACKEND_PORT=8001

# Label Frontend Configuration
LABEL_FRONTEND_PORT=3345

# API URL for Frontend
VITE_API_URL=http://localhost:8001/api/v1
```

### Bước 3: Khởi động lại Docker services

```bash
# Stop các services hiện tại
docker-compose down

# Start lại với config mới
docker-compose up -d --build
```

### Bước 4: Kiểm tra services đang chạy

```bash
# Xem tất cả containers
docker-compose ps

# Bạn sẽ thấy:
# - postgres (port 5432)
# - sentiment-service (port 8002)
# - embedding-service (port 8003)  ← MỚI
# - label-backend (port 8001)
# - label-frontend (port 3345)
```

### Bước 5: Test embedding service

```bash
# Test health endpoint
curl http://localhost:8003/api/v1/health

# Nếu OK, bạn sẽ thấy response:
# {"status":"healthy","model_loaded":true,...}
```

## Lỗi Khác: "invalid proto"

Nếu bạn thấy lỗi `invalid proto:`, có thể do:

1. **Port bị trùng** - Check xem port 8003 đã được dùng chưa:
   ```bash
   # Windows
   netstat -ano | findstr :8003
   
   # Linux/Mac
   lsof -i :8003
   ```
   
   Nếu bị trùng, đổi port trong `.env`:
   ```bash
   # Dùng port khác còn trống (ví dụ 8004, 8005, etc.)
   EMBEDDING_EXTERNAL_PORT=8004
   ```

2. **Docker network issue** - Rebuild lại:
   ```bash
   docker-compose down --volumes
   docker-compose up -d --build
   ```

## Seed Data

Sau khi tất cả services đã chạy OK:

```bash
# Cài đặt requests nếu chưa có
pip install requests

# Chạy script seed data
python seed_data.py
```

## Troubleshooting

### Services không start được

```bash
# Xem logs để debug
docker-compose logs embedding-service
docker-compose logs label-backend

# Xem logs real-time
docker-compose logs -f
```

### Port conflict

Nếu port bị trùng, sửa trong `.env`:

```bash
# Dùng port khác còn trống
EMBEDDING_EXTERNAL_PORT=8004
```

Sau đó restart:
```bash
docker-compose down
docker-compose up -d --build
```

### Model không load được

Check xem folder model có đúng không:

```bash
# Kiểm tra structure
ls -la bge-m3-finetuned-transformer/
# Phải có: vn_embedding_bgem3/

ls -la bge-m3-finetuned-transformer/vn_embedding_bgem3/
# Phải có: model.onnx, tokenizer.json, config.json, etc.
```

## Summary

**TL;DR - Thêm vào file `.env`:**

```bash
# Port 8003 vì 8002 đã dùng cho sentiment
EMBEDDING_EXTERNAL_PORT=8003
```

Sau đó:

```bash
docker-compose down
docker-compose up -d --build
python seed_data.py
```

Done! 🎉

