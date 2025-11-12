# Hướng dẫn Build và Deploy

## Yêu cầu
- Docker và Docker Compose
- Git

## Bước 1: Chuẩn bị file .env

Tạo file `.env` trong thư mục root với nội dung sau:

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
VITE_API_URL=http://localhost:8001/api/v1

# Sentiment Service
SENTIMENT_PORT=8000
SENTIMENT_MODEL_PATH=/models/sentiments
```

## Bước 2: Build và chạy các services

### Build tất cả services:
```bash
docker-compose build
```

### Chạy tất cả services:
```bash
docker-compose up -d
```

### Xem logs:
```bash
# Tất cả services
docker-compose logs -f

# Một service cụ thể
docker-compose logs -f label-backend
docker-compose logs -f sentiment-service
docker-compose logs -f label-frontend
```

## Bước 3: Kiểm tra services

### Kiểm tra trạng thái:
```bash
docker-compose ps
```

### Health checks:
- Sentiment Service: http://localhost:8000/api/v1/health
- Label Backend: http://localhost:8001/api/v1/health
- Frontend: http://localhost:3000

### API Documentation:
- Sentiment Service API: http://localhost:8000/docs
- Label Backend API: http://localhost:8001/docs

## Bước 4: Truy cập ứng dụng

Mở trình duyệt và truy cập: **http://localhost:3000**

## Kiến trúc Services

```
┌─────────────────┐
│   Frontend      │
│   (Port 3000)   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐      ┌──────────────────┐
│  Label Backend  │─────→│ Sentiment Service│
│   (Port 8001)   │      │   (Port 8000)    │
└────────┬────────┘      └──────────────────┘
         │
         ↓
┌─────────────────┐
│   PostgreSQL    │
│   (Port 5432)   │
└─────────────────┘
```

## Các lệnh hữu ích

### Dừng tất cả services:
```bash
docker-compose down
```

### Xóa volumes (reset database):
```bash
docker-compose down -v
```

### Rebuild một service cụ thể:
```bash
docker-compose build label-backend
docker-compose up -d label-backend
```

### Restart một service:
```bash
docker-compose restart label-backend
```

### Exec vào container:
```bash
docker-compose exec label-backend bash
docker-compose exec sentiment-service bash
```

## Troubleshooting

### Lỗi: "All connection attempts failed"
- Kiểm tra sentiment-service đã chạy chưa: `docker-compose ps sentiment-service`
- Kiểm tra logs: `docker-compose logs sentiment-service`
- Đợi sentiment service khởi động xong (có thể mất 30-60s do load model)

### Database connection error
- Kiểm tra postgres đã chạy: `docker-compose ps postgres`
- Kiểm tra credentials trong file .env
- Reset database: `docker-compose down -v && docker-compose up -d`

### Frontend không load được
- Kiểm tra VITE_API_URL trong .env
- Rebuild frontend: `docker-compose build label-frontend && docker-compose up -d label-frontend`

### Port đã được sử dụng
- Thay đổi port trong file .env
- Ví dụ: `LABEL_BACKEND_PORT=8002`

## Development Mode

Để chạy frontend ở development mode với hot reload:

1. Comment service `label-frontend` trong docker-compose.yml
2. Uncomment service `label-frontend-dev`
3. Chạy: `docker-compose up -d label-frontend-dev`
4. Truy cập: http://localhost:5173

## Tính năng mới: Sentiment Analysis

Hệ thống đã được tích hợp tính năng phân tích sentiment cho feedback:

1. **Submit Feedback**: Gửi feedback từ khách hàng qua giao diện
2. **Auto Analysis**: Tự động phân tích sentiment (4 labels):
   - POSITIVE (Tích cực)
   - NEGATIVE (Tiêu cực)
   - EXTREMELY_NEGATIVE (Rất tiêu cực)
   - NEUTRAL (Trung tính)
3. **Save to DB**: Lưu feedback + kết quả phân tích vào database
4. **View & Filter**: Xem danh sách, filter theo sentiment/source, pagination

### Nguồn feedback hỗ trợ:
- Web
- App
- Map
- Form khảo sát
- Tổng đài

## Cấu trúc Database

### Table: labels (đã có)
Quản lý hierarchical labels (3 levels)

### Table: feedback_sentiments (mới)
Lưu trữ feedback và kết quả sentiment analysis
- Tự động tạo khi khởi động label-backend
- Không cần migration thủ công



