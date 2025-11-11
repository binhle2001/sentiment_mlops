import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Input,
  Select,
  Button,
  Table,
  Tag,
  Space,
  message,
  Typography,
  Row,
  Col,
  Divider,
  Statistic,
} from 'antd';
import {
  SmileOutlined,
  FrownOutlined,
  MehOutlined,
  WarningOutlined,
  SendOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { feedbackAPI } from '../services/api';

const { TextArea } = Input;
const { Title, Text } = Typography;
const { Option } = Select;

const FeedbackSentiment = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [feedbacks, setFeedbacks] = useState([]);
  const [total, setTotal] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [filters, setFilters] = useState({});
  const [latestResult, setLatestResult] = useState(null);

  // Fetch feedbacks on mount and when pagination/filters change
  useEffect(() => {
    fetchFeedbacks();
  }, [currentPage, pageSize, filters]);

  const fetchFeedbacks = async () => {
    try {
      setLoading(true);
      const params = {
        skip: (currentPage - 1) * pageSize,
        limit: pageSize,
        ...filters,
      };
      const response = await feedbackAPI.getFeedbacks(params);
      setFeedbacks(response.feedbacks);
      setTotal(response.total);
    } catch (error) {
      message.error('Không thể tải danh sách feedback');
      console.error('Error fetching feedbacks:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (values) => {
    try {
      setLoading(true);
      const response = await feedbackAPI.submitFeedback(
        values.feedback_text,
        values.feedback_source
      );
      
      setLatestResult(response);
      
      if (response.level1_id) {
        message.success('Phân tích sentiment và intent thành công!');
      } else {
        message.success('Phân tích sentiment thành công! (Intent không khả dụng)');
      }
      
      form.resetFields();
      
      // Refresh the list
      setCurrentPage(1);
      fetchFeedbacks();
    } catch (error) {
      message.error('Không thể phân tích feedback. Vui lòng thử lại.');
      console.error('Error submitting feedback:', error);
    } finally {
      setLoading(false);
    }
  };


  const getSentimentColor = (label) => {
    switch (label) {
      case 'POSITIVE':
        return 'success';
      case 'NEGATIVE':
        return 'error';
      case 'EXTREMELY_NEGATIVE':
        return 'error';
      case 'NEUTRAL':
        return 'default';
      default:
        return 'default';
    }
  };

  const getSentimentIcon = (label) => {
    switch (label) {
      case 'POSITIVE':
        return <SmileOutlined />;
      case 'NEGATIVE':
        return <FrownOutlined />;
      case 'EXTREMELY_NEGATIVE':
        return <WarningOutlined />;
      case 'NEUTRAL':
        return <MehOutlined />;
      default:
        return null;
    }
  };

  const getSentimentText = (label) => {
    switch (label) {
      case 'POSITIVE':
        return 'Tích cực';
      case 'NEGATIVE':
        return 'Tiêu cực';
      case 'EXTREMELY_NEGATIVE':
        return 'Rất tiêu cực';
      case 'NEUTRAL':
        return 'Trung tính';
      default:
        return label;
    }
  };

  const getSourceText = (source) => {
    const sourceMap = {
      'web': 'Web',
      'app': 'App',
      'map': 'Map',
      'form khảo sát': 'Form khảo sát',
      'tổng đài': 'Tổng đài',
    };
    return sourceMap[source] || source;
  };

  const columns = [
    {
      title: 'Nội dung Feedback',
      dataIndex: 'feedback_text',
      key: 'feedback_text',
      width: '30%',
      ellipsis: true,
      render: (text) => (
        <Text ellipsis={{ tooltip: text }} style={{ maxWidth: 300 }}>
          {text}
        </Text>
      ),
    },
    {
      title: 'Sentiment',
      dataIndex: 'sentiment_label',
      key: 'sentiment_label',
      width: '12%',
      filters: [
        { text: 'Tích cực', value: 'POSITIVE' },
        { text: 'Tiêu cực', value: 'NEGATIVE' },
        { text: 'Rất tiêu cực', value: 'EXTREMELY_NEGATIVE' },
        { text: 'Trung tính', value: 'NEUTRAL' },
      ],
      render: (label) => (
        <Tag color={getSentimentColor(label)} icon={getSentimentIcon(label)}>
          {getSentimentText(label)}
        </Tag>
      ),
    },
    {
      title: 'Intent Classification',
      key: 'intent',
      width: '30%',
      render: (text, record) => {
        if (record.level1_id) {
          return (
            <Space direction="vertical" size={2}>
              <div>
                <Tag color="blue" style={{ fontSize: '11px' }}>
                  {record.level1_name}
                </Tag>
                <span style={{ fontSize: '11px' }}>→</span>
                <Tag color="cyan" style={{ fontSize: '11px' }}>
                  {record.level2_name}
                </Tag>
                <span style={{ fontSize: '11px' }}>→</span>
                <Tag color="geekblue" style={{ fontSize: '11px' }}>
                  {record.level3_name}
                </Tag>
              </div>
            </Space>
          );
        }
        return <Text type="secondary" style={{ fontSize: '11px' }}>Chưa phân loại</Text>;
      },
    },
    {
      title: 'Nguồn',
      dataIndex: 'feedback_source',
      key: 'feedback_source',
      width: '10%',
      filters: [
        { text: 'Web', value: 'web' },
        { text: 'App', value: 'app' },
        { text: 'Map', value: 'map' },
        { text: 'Form khảo sát', value: 'form khảo sát' },
        { text: 'Tổng đài', value: 'tổng đài' },
      ],
      render: (source) => <Tag>{getSourceText(source)}</Tag>,
    },
    {
      title: 'Thời gian',
      dataIndex: 'created_at',
      key: 'created_at',
      width: '18%',
      render: (date) => new Date(date).toLocaleString('vi-VN'),
      sorter: (a, b) => new Date(a.created_at) - new Date(b.created_at),
    },
  ];

  const handleTableChange = (pagination, filters) => {
    setCurrentPage(pagination.current);
    setPageSize(pagination.pageSize);
    
    const newFilters = {};
    if (filters.sentiment_label && filters.sentiment_label.length > 0) {
      newFilters.sentiment_label = filters.sentiment_label[0];
    }
    if (filters.feedback_source && filters.feedback_source.length > 0) {
      newFilters.feedback_source = filters.feedback_source[0];
    }
    setFilters(newFilters);
  };

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <Title level={2}>Phân tích Sentiment Feedback</Title>
      
      {/* Submit Feedback Form */}
      <Card title="Gửi Feedback mới" style={{ marginBottom: 24 }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{ feedback_source: 'web' }}
        >
          <Row gutter={16}>
            <Col xs={24} lg={18}>
              <Form.Item
                label="Nội dung Feedback"
                name="feedback_text"
                rules={[
                  { required: true, message: 'Vui lòng nhập nội dung feedback' },
                  { min: 1, message: 'Nội dung không được để trống' },
                ]}
              >
                <TextArea
                  rows={4}
                  placeholder="Nhập nội dung feedback từ khách hàng..."
                  maxLength={5000}
                  showCount
                />
              </Form.Item>
            </Col>
            <Col xs={24} lg={6}>
              <Form.Item
                label="Nguồn Feedback"
                name="feedback_source"
                rules={[{ required: true, message: 'Vui lòng chọn nguồn feedback' }]}
              >
                <Select placeholder="Chọn nguồn">
                  <Option value="web">Web</Option>
                  <Option value="app">App</Option>
                  <Option value="map">Map</Option>
                  <Option value="form khảo sát">Form khảo sát</Option>
                  <Option value="tổng đài">Tổng đài</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
                icon={<SendOutlined />}
              >
                Phân tích Sentiment
              </Button>
              <Button onClick={() => form.resetFields()}>Xóa</Button>
            </Space>
          </Form.Item>
        </Form>

        {/* Latest Result Display */}
        {latestResult && (
          <>
            <Divider>Kết quả phân tích mới nhất</Divider>
            <Row gutter={16}>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="Sentiment"
                    value={getSentimentText(latestResult.sentiment_label)}
                    prefix={getSentimentIcon(latestResult.sentiment_label)}
                    valueStyle={{
                      color:
                        latestResult.sentiment_label === 'POSITIVE'
                          ? '#3f8600'
                          : latestResult.sentiment_label === 'NEGATIVE' ||
                            latestResult.sentiment_label === 'EXTREMELY_NEGATIVE'
                          ? '#cf1322'
                          : '#666',
                    }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="Độ tin cậy"
                    value={(latestResult.confidence_score * 100).toFixed(2)}
                    suffix="%"
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="Nguồn"
                    value={getSourceText(latestResult.feedback_source)}
                  />
                </Card>
              </Col>
            </Row>

            {/* Intent Classification Result (by Gemini AI) */}
            {latestResult.level1_id && (
              <>
                <Divider>Phân loại Intent (Gemini AI)</Divider>
                <Card style={{ background: '#f6ffed', borderColor: '#b7eb8f' }}>
                  <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                    <div>
                      <Text strong style={{ fontSize: '16px', marginRight: '8px' }}>
                        Intent được chọn:
                      </Text>
                      <Tag color="blue" style={{ fontSize: '14px', padding: '4px 12px' }}>
                        {latestResult.level1_name}
                      </Tag>
                      <span style={{ margin: '0 8px', fontSize: '16px' }}>→</span>
                      <Tag color="cyan" style={{ fontSize: '14px', padding: '4px 12px' }}>
                        {latestResult.level2_name}
                      </Tag>
                      <span style={{ margin: '0 8px', fontSize: '16px' }}>→</span>
                      <Tag color="geekblue" style={{ fontSize: '14px', padding: '4px 12px' }}>
                        {latestResult.level3_name}
                      </Tag>
                    </div>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      ✨ Được chọn tự động bởi Gemini AI từ các ứng viên có độ tương đồng cao nhất
                    </Text>
                  </Space>
                </Card>
              </>
            )}
          </>
        )}
      </Card>

      {/* Feedbacks List */}
      <Card
        title="Danh sách Feedback"
        extra={
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchFeedbacks}
            loading={loading}
          >
            Làm mới
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={feedbacks}
          loading={loading}
          rowKey="id"
          pagination={{
            current: currentPage,
            pageSize: pageSize,
            total: total,
            showSizeChanger: true,
            showTotal: (total) => `Tổng ${total} feedback`,
            pageSizeOptions: ['10', '20', '50', '100'],
          }}
          onChange={handleTableChange}
          scroll={{ x: 1000 }}
        />
      </Card>
    </div>
  );
};

export default FeedbackSentiment;

