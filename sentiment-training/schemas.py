from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema cho endpoint health check."""

    status: str = Field(..., description="Trạng thái service")
    message: Optional[str] = Field(default=None, description="Thông báo chi tiết")


class TrainingTriggerRequest(BaseModel):
    """Yêu cầu kích hoạt huấn luyện sentiment."""

    force: bool = Field(
        default=False,
        description="Nếu true sẽ bỏ qua ngưỡng 20% dữ liệu thay đổi và chạy ngay",
    )


class TrainingTriggerResponse(BaseModel):
    """Kết quả trả về sau khi trigger huấn luyện."""

    triggered: bool = Field(..., description="Cho biết pipeline có được kích hoạt hay không")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Thông tin chi tiết (tỷ lệ thay đổi, run_id, báo cáo, ...)",
    )



