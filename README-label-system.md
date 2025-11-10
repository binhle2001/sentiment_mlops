# Label Management System

Hệ thống quản lý label phân cấp 3 levels cho MLOps pipeline phân loại feedback khách hàng.

## Tổng quan

Hệ thống bao gồm:
- **PostgreSQL**: Database lưu trữ labels hierarchy
- **FastAPI Backend**: REST API service quản lý labels
- **React Frontend**: Web UI để quản lý labels

## Kiến trúc

```
┌─────────────────┐
│  React Frontend │ :3000
│   (Nginx)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FastAPI Backend │ :8001
│  label-service  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │ :5432
│   label_db      │
└─────────────────┘
```

## Hierarchy Structure

- **Level 1**: Top level categories (e.g., Dịch vụ, Sản phẩm, Hỗ trợ)
  - **Level 2**: Sub-categories (e.g., Giao dịch, Thanh toán, Tư vấn)
    - **Level 3**: Specific labels (e.g., Chuyển tiền, Rút tiền, Thanh toán hóa đơn)

## Yêu cầu hệ thống

- Docker & Docker Compose
- 2GB RAM trở lên
- 5GB disk space

## Cài đặt và Chạy

### 1. Clone và chuẩn bị

```bash
cd D:\binhltl_code\cxm_bidv_mlops
```

### 2. Cấu hình Environment Variables (tùy chọn)

File `.env` đã được tạo sẵn với các giá trị mặc định. Bạn có thể chỉnh sửa nếu cần:

```bash
# PostgreSQL
POSTGRES_USER=labeluser
POSTGRES_PASSWORD=labelpass123
POSTGRES_DB=label_db

# Backend
LABEL_BACKEND_PORT=8001

# Frontend
LABEL_FRONTEND_PORT=3000
```

### 3. Build và Start Services

```bash
# Build tất cả services
docker-compose build

# Start services
docker-compose up -d

# Hoặc kết hợp: build và start
docker-compose up -d --build
```

### 4. Kiểm tra Services

```bash
# Xem logs
docker-compose logs -f

# Kiểm tra health
curl http://localhost:8001/api/v1/health
```

## Truy cập Hệ thống

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## Sử dụng

### Tạo Label mới

1. Mở UI tại http://localhost:3000
2. Click "Create Label"
3. Chọn Level (1, 2, hoặc 3)
4. Nếu Level 2 hoặc 3, chọn parent label
5. Nhập tên và mô tả
6. Click OK

### Sửa Label

1. Click icon "Edit" trên label muốn sửa
2. Cập nhật tên hoặc mô tả
3. Click OK

### Xóa Label

1. Click icon "Delete" trên label muốn xóa
2. Xác nhận xóa (lưu ý: sẽ xóa tất cả children)

### Thêm Child Label

1. Click icon "+" trên parent label
2. Form sẽ tự động chọn level và parent
3. Nhập tên và mô tả
4. Click OK

## API Endpoints

### Labels

- `GET /api/v1/labels` - Lấy danh sách labels (có filter)
- `GET /api/v1/labels/tree` - Lấy cây hierarchy đầy đủ
- `GET /api/v1/labels/{id}` - Lấy label theo ID
- `GET /api/v1/labels/{id}/children` - Lấy children của label
- `POST /api/v1/labels` - Tạo label mới
- `PUT /api/v1/labels/{id}` - Cập nhật label
- `DELETE /api/v1/labels/{id}` - Xóa label

### Health Check

- `GET /api/v1/health` - Kiểm tra health của service

## Database Schema

```sql
CREATE TABLE labels (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    level INTEGER CHECK (level IN (1, 2, 3)),
    parent_id UUID REFERENCES labels(id) ON DELETE CASCADE,
    description TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    UNIQUE(name, parent_id)
);
```

## Dữ liệu Mẫu

Hệ thống đã được cấu hình với dữ liệu mẫu sẵn:

**Level 1:**
- Dịch vụ
- Sản phẩm
- Hỗ trợ

**Level 2 (under Dịch vụ):**
- Giao dịch
- Thanh toán

**Level 3 (under Giao dịch):**
- Chuyển tiền
- Rút tiền

**Level 3 (under Thanh toán):**
- Thanh toán hóa đơn
- Thanh toán online

## Quản lý Services

### Xem Logs

```bash
# Tất cả services
docker-compose logs -f

# Một service cụ thể
docker-compose logs -f label-backend
docker-compose logs -f label-frontend
docker-compose logs -f postgres
```

### Stop Services

```bash
docker-compose stop
```

### Start lại Services

```bash
docker-compose start
```

### Restart Services

```bash
docker-compose restart
```

### Xóa Services

```bash
# Stop và xóa containers
docker-compose down

# Xóa cả volumes (database data)
docker-compose down -v
```

### Rebuild Services

```bash
# Rebuild một service
docker-compose build label-backend

# Rebuild và restart
docker-compose up -d --build label-backend
```

## Troubleshooting

### Backend không kết nối được Database

```bash
# Kiểm tra PostgreSQL
docker-compose logs postgres

# Restart backend
docker-compose restart label-backend
```

### Frontend không load được

```bash
# Kiểm tra logs
docker-compose logs label-frontend

# Rebuild frontend
docker-compose up -d --build label-frontend
```

### Port conflicts

Nếu port đã được sử dụng, chỉnh sửa file `.env`:

```bash
LABEL_BACKEND_PORT=8002  # thay vì 8001
LABEL_FRONTEND_PORT=3001  # thay vì 3000
```

Sau đó restart:

```bash
docker-compose down
docker-compose up -d
```

## Development

### Backend Development

```bash
cd label-service

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py
```

### Frontend Development

```bash
cd label-frontend

# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build
```

### Database Migrations (Alembic)

```bash
cd label-service

# Tạo migration mới
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Architecture Decisions

### Backend
- **FastAPI**: Modern, fast, async-first framework
- **SQLAlchemy Async**: Async ORM for better performance
- **PostgreSQL**: Robust, supports recursive queries
- **Pydantic**: Data validation và settings management

### Frontend
- **React + Vite**: Fast development và build
- **Ant Design**: Professional UI components
- **Axios**: HTTP client với interceptors

### Deployment
- **Docker Multi-stage builds**: Smaller images
- **Nginx**: Static file serving và reverse proxy
- **Health checks**: Auto-healing containers

## Performance

- Backend: ~100-500ms response time
- Frontend: Instant UI updates
- Database: Indexed queries, connection pooling
- Docker: Resource limits configured

## Security

- Non-root users in containers
- SQL injection prevention (SQLAlchemy)
- CORS configured
- Input validation (Pydantic)
- Cascade delete protection

## Next Steps

1. Tích hợp với Embedding service
2. Tích hợp với Sentiment service
3. API authentication/authorization
4. Label versioning
5. Export/Import labels
6. Analytics dashboard

## Support

Nếu gặp vấn đề, kiểm tra:
1. Docker logs: `docker-compose logs`
2. Service health: http://localhost:8001/api/v1/health
3. Database connection: `docker-compose exec postgres psql -U labeluser -d label_db`

## License

Internal use for CXM BIDV MLOps project.



