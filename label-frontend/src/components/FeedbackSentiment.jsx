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
  const [topIntents, setTopIntents] = useState([]);
  const [loadingIntents, setLoadingIntents] = useState(false);

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
      message.success('Phân tích sentiment thành công!');
      form.resetFields();
      
      // Get intent analysis for the feedback
      fetchIntentsForFeedback(response.id);
      
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

  const fetchIntentsForFeedback = async (feedbackId) => {
    try {
      setLoadingIntents(true);
      const response = await feedbackAPI.getIntentsForFeedback(feedbackId);
      setTopIntents(response.intents || []);
      if (response.intents && response.intents.length > 0) {
        message.success(`Phân tích intent thành công! Tìm thấy ${response.intents.length} intent phù hợp.`);
      }
    } catch (error) {
      message.warning('Không thể phân tích intent. Có thể chưa có đủ dữ liệu label.');
      console.error('Error fetching intents:', error);
      setTopIntents([]);
    } finally {
      setLoadingIntents(false);
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
      width: '40%',
      ellipsis: true,
      render: (text) => (
        <Text ellipsis={{ tooltip: text }} style={{ maxWidth: 400 }}>
          {text}
        </Text>
      ),
    },
    {
      title: 'Sentiment',
      dataIndex: 'sentiment_label',
      key: 'sentiment_label',
      width: '15%',
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
      title: 'Độ tin cậy',
      dataIndex: 'confidence_score',
      key: 'confidence_score',
      width: '12%',
      render: (score) => `${(score * 100).toFixed(2)}%`,
      sorter: (a, b) => a.confidence_score - b.confidence_score,
    },
    {
      title: 'Nguồn',
      dataIndex: 'feedback_source',
      key: 'feedback_source',
      width: '13%',
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
      width: '20%',
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

            {/* Intent Analysis Results */}
            <Divider>Top 50 Intent Triplets (Độ tương đồng cao nhất)</Divider>
            {loadingIntents ? (
              <div style={{ textAlign: 'center', padding: '20px' }}>
                <Space direction="vertical">
                  <ReloadOutlined spin style={{ fontSize: '24px' }} />
                  <Text>Đang phân tích intent...</Text>
                </Space>
              </div>
            ) : topIntents.length > 0 ? (
              <Table
                dataSource={topIntents}
                pagination={false}
                size="small"
                rowKey={(record, index) => `${record.level1.id}-${record.level2.id}-${record.level3.id}-${index}`}
                columns={[
                  {
                    title: '#',
                    key: 'index',
                    width: '5%',
                    render: (text, record, index) => index + 1,
                  },
                  {
                    title: 'Intent Path',
                    key: 'intent_path',
                    width: '75%',
                    render: (text, record) => (
                      <Text>
                        <Tag color="blue">{record.level1.name}</Tag>
                        <span style={{ margin: '0 8px' }}>→</span>
                        <Tag color="cyan">{record.level2.name}</Tag>
                        <span style={{ margin: '0 8px' }}>→</span>
                        <Tag color="geekblue">{record.level3.name}</Tag>
                      </Text>
                    ),
                  },
                  {
                    title: 'Độ tương đồng',
                    key: 'similarity',
                    width: '20%',
                    align: 'center',
                    render: (text, record) => (
                      <Tag color={record.avg_cosine_similarity >= 0.7 ? 'green' : record.avg_cosine_similarity >= 0.5 ? 'orange' : 'default'}>
                        {(record.avg_cosine_similarity * 100).toFixed(2)}%
                      </Tag>
                    ),
                    sorter: (a, b) => a.avg_cosine_similarity - b.avg_cosine_similarity,
                  },
                ]}
              />
            ) : (
              <div style={{ textAlign: 'center', padding: '20px' }}>
                <Text type="secondary">
                  Không có intent nào được tìm thấy. Vui lòng đảm bảo labels đã có embedding.
                </Text>
              </div>
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

