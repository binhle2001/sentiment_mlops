# Hướng dẫn Deploy nhanh

## Bước 1: Tạo file .env

Copy file `.env.template` thành `.env`:

```bash
copy .env.template .env
```

Hoặc tạo file `.env` với nội dung:

```env
# PostgreSQL Configuration
POSTGRES_USER=labeluser
POSTGRES_PASSWORD=labelpass123
POSTGRES_DB=label_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Label Backend Service
LABEL_BACKEND_PORT=8001

# Label Frontend Service  
LABEL_FRONTEND_PORT=3000
VITE_API_URL=http://103.167.84.162:8001/api/v1

# Sentiment Service
# SENTIMENT_EXTERNAL_PORT: Port expose ra ngoài (dùng 8002 vì 8000 đã bận)
SENTIMENT_EXTERNAL_PORT=8002
# SENTIMENT_PORT: Port bên trong container mà sentiment service lắng nghe
SENTIMENT_PORT=8005
# SENTIMENT_HOST: Để 0.0.0.0 để service có thể nhận request từ docker network
SENTIMENT_HOST=0.0.0.0
```

## Bước 2: Build và chạy

```bash
docker-compose up -d --build
```

## Bước 3: Kiểm tra

```bash
# Xem trạng thái
docker-compose ps

# Xem logs
docker-compose logs -f
```

## Truy cập

- **Frontend**: http://103.167.84.162:3000
- **Label Backend API**: http://103.167.84.162:8001/docs
- **Sentiment Service API**: http://103.167.84.162:8002/docs

## Các lệnh hữu ích

```bash
# Stop tất cả
docker-compose down

# Restart
docker-compose restart

# Rebuild một service
docker-compose up -d --build sentiment-service

# Xem logs của một service
docker-compose logs -f sentiment-service
```

## Thay đổi port

Muốn đổi port? Chỉnh trong file `.env`:

```env
SENTIMENT_PORT=8003  # Đổi từ 8002 sang 8003
LABEL_BACKEND_PORT=8005  # Đổi backend port
```

Sau đó:

```bash
docker-compose down
docker-compose up -d
```

## Cấu trúc services

```
Frontend (Port 3000)
    ↓ gọi API
Label Backend (Port 8001) ----→ Sentiment Service (Port 8002)
    ↓                                     ↓
PostgreSQL (Port 5432)            Sentiment Model
```

**Lưu ý**: Các services trong Docker gọi nhau bằng tên service (vd: `sentiment-service:8000`), không phải localhost.

