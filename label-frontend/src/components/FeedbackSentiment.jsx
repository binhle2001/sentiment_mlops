import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Input,
  Select,
  Button,
  Modal,
  Table,
  Tag,
  Space,
  Upload,
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
  CheckOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import { feedbackAPI, labelAPI } from '../services/api';

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
  const [editForm] = Form.useForm();
  const [labelTree, setLabelTree] = useState([]);
  const [availableLevel2, setAvailableLevel2] = useState([]);
  const [availableLevel3, setAvailableLevel3] = useState([]);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [editingFeedback, setEditingFeedback] = useState(null);
  const [selectedLevel1, setSelectedLevel1] = useState(null);
  const [selectedLevel2, setSelectedLevel2] = useState(null);
  const [selectedLevel3, setSelectedLevel3] = useState(null);
  const [updatingFeedback, setUpdatingFeedback] = useState(false);
  const [labelLoading, setLabelLoading] = useState(false);
  const [confirmingId, setConfirmingId] = useState(null);
  const [importing, setImporting] = useState(false);
  const [importingSimple, setImportingSimple] = useState(false);

  const loadLabelTree = async () => {
    try {
      setLabelLoading(true);
      const data = await labelAPI.getTree();
      const treeData = Array.isArray(data) ? data : [];
      setLabelTree(treeData);
      return treeData;
    } catch (error) {
      message.error('Không thể tải danh sách intent');
      console.error('Error fetching label tree:', error);
      return [];
    } finally {
      setLabelLoading(false);
    }
  };

  useEffect(() => {
    loadLabelTree();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  const handleEdit = async (record) => {
    setEditingFeedback(record);

    let tree = labelTree;
    if (!tree || tree.length === 0) {
      tree = await loadLabelTree();
    }

    setSelectedLevel1(record.level1_id || null);
    setSelectedLevel2(record.level2_id || null);
    setSelectedLevel3(record.level3_id || null);

    const level1Node = (tree || []).find((node) => node.id === record.level1_id) || null;
    setAvailableLevel2(level1Node?.children || []);

    const level2Node =
      level1Node?.children?.find((child) => child.id === record.level2_id) || null;
    setAvailableLevel3(level2Node?.children || []);

    editForm.setFieldsValue({
      sentiment_label: record.sentiment_label,
      level1_id: record.level1_id || undefined,
      level2_id: record.level2_id || undefined,
      level3_id: record.level3_id || undefined,
    });

    setEditModalVisible(true);
  };

  const handleConfirm = async (record) => {
    if (!record?.id) {
      return;
    }
    setConfirmingId(record.id);
    try {
      const updated = await feedbackAPI.confirmFeedback(record.id);
      message.success('Đã xác nhận feedback!');
      setFeedbacks((prev) =>
        prev.map((item) => (item.id === updated.id ? { ...item, ...updated } : item))
      );
      if (latestResult && updated?.id === latestResult.id) {
        setLatestResult(updated);
      }
    } catch (error) {
      const detail = error?.response?.data?.detail || 'Không thể xác nhận feedback';
      message.error(detail);
      console.error('Error confirming feedback:', error);
    } finally {
      setConfirmingId(null);
    }
  };

  const handleLevel1Change = (value) => {
    const normalized = value || null;
    setSelectedLevel1(normalized);

    const level1Node = (labelTree || []).find((node) => node.id === value) || null;
    setAvailableLevel2(level1Node?.children || []);
    setAvailableLevel3([]);
    setSelectedLevel2(null);
    setSelectedLevel3(null);
    editForm.setFieldsValue({
      level2_id: undefined,
      level3_id: undefined,
    });
  };

  const handleLevel2Change = (value) => {
    const normalized = value || null;
    setSelectedLevel2(normalized);

    const level2Node = (availableLevel2 || []).find((node) => node.id === value) || null;
    setAvailableLevel3(level2Node?.children || []);
    setSelectedLevel3(null);
    editForm.setFieldsValue({
      level3_id: undefined,
    });
  };

  const handleLevel3Change = (value) => {
    setSelectedLevel3(value || null);
  };

  const handleEditCancel = () => {
    setEditModalVisible(false);
    setEditingFeedback(null);
    editForm.resetFields();
    setAvailableLevel2([]);
    setAvailableLevel3([]);
    setSelectedLevel1(null);
    setSelectedLevel2(null);
    setSelectedLevel3(null);
  };

  const handleEditSubmit = async () => {
    try {
      const values = await editForm.validateFields();
      if (!editingFeedback) {
        return;
      }

      const payload = {};
      const currentLevel1 = editingFeedback.level1_id || null;
      const currentLevel2 = editingFeedback.level2_id || null;
      const currentLevel3 = editingFeedback.level3_id || null;

      if (
        values.sentiment_label &&
        values.sentiment_label !== editingFeedback.sentiment_label
      ) {
        payload.sentiment_label = values.sentiment_label;
      }

      if (selectedLevel1 !== currentLevel1) {
        payload.level1_id = selectedLevel1;
      }

      if (
        selectedLevel2 !== currentLevel2 ||
        (selectedLevel1 === null && currentLevel2 !== null) ||
        (selectedLevel1 && selectedLevel2 === null && currentLevel2 !== null)
      ) {
        payload.level2_id = selectedLevel1 ? selectedLevel2 : null;
      }

      if (
        selectedLevel3 !== currentLevel3 ||
        (selectedLevel2 === null && currentLevel3 !== null)
      ) {
        payload.level3_id = selectedLevel2 ? selectedLevel3 : null;
      }

      if (Object.keys(payload).length === 0) {
        message.info('Không có thay đổi nào để cập nhật.');
        return;
      }

      setUpdatingFeedback(true);
      const updated = await feedbackAPI.updateFeedback(editingFeedback.id, payload);
      message.success('Cập nhật feedback thành công!');
      await fetchFeedbacks();
      if (latestResult && updated?.id === latestResult.id) {
        setLatestResult(updated);
      }
      handleEditCancel();
    } catch (error) {
      if (error?.errorFields) {
        return;
      }
      const detail = error.response?.data?.detail || 'Không thể cập nhật feedback';
      message.error(detail);
      console.error('Error updating feedback:', error);
    } finally {
      setUpdatingFeedback(false);
    }
  };

  const handleImportExcel = async (file) => {
    if (!file) {
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setImporting(true);
      const result = await feedbackAPI.importFeedbacks(formData);
      message.success(`Đã import ${result.imported} feedback thành công.`);
      if (result.failed > 0) {
        const warningMessage = result.log_file
          ? `Có ${result.failed} dòng lỗi. Xem log tại ${result.log_file}`
          : `Có ${result.failed} dòng lỗi.`;
        message.warning(warningMessage);
      }
      await fetchFeedbacks();
    } catch (error) {
      const detail = error?.response?.data?.detail || 'Không thể import dữ liệu feedback';
      message.error(detail);
      console.error('Error importing feedbacks:', error);
    } finally {
      setImporting(false);
    }
  };

  const uploadProps = {
    accept: '.xlsx',
    showUploadList: false,
    beforeUpload: (file) => {
      handleImportExcel(file);
      return false;
    },
  };

  const handleImportExcelSimple = async (file) => {
    if (!file) {
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setImportingSimple(true);
      const result = await feedbackAPI.importFeedbacksSimple(formData);
      message.success(`Đã import ${result.imported} feedback thành công và tự động phân tích sentiment/intent.`);
      if (result.failed > 0) {
        const warningMessage = result.log_file
          ? `Có ${result.failed} dòng lỗi. Xem log tại ${result.log_file}`
          : `Có ${result.failed} dòng lỗi.`;
        message.warning(warningMessage);
      }
      await fetchFeedbacks();
    } catch (error) {
      const detail = error?.response?.data?.detail || 'Không thể import dữ liệu feedback';
      message.error(detail);
      console.error('Error importing feedbacks:', error);
    } finally {
      setImportingSimple(false);
    }
  };

  const uploadSimpleProps = {
    accept: '.xlsx',
    showUploadList: false,
    beforeUpload: (file) => {
      handleImportExcelSimple(file);
      return false;
    },
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
    {
      title: 'Xác nhận',
      dataIndex: 'is_model_confirmed',
      key: 'is_model_confirmed',
      width: '10%',
      render: (value) => (
        <Tag color={value ? 'green' : 'default'}>{value ? 'Đã xác nhận' : 'Chưa'}</Tag>
      ),
    },
    {
      title: 'Thao tác',
      key: 'actions',
      width: '14%',
      render: (_, record) => {
        const canConfirm = !!record.level1_id && !record.is_model_confirmed;
        return (
          <Space>
            {canConfirm && (
              <Button
                type="link"
                icon={<CheckOutlined />}
                onClick={() => handleConfirm(record)}
                loading={confirmingId === record.id}
              >
                Xác nhận
              </Button>
            )}
            <Button type="link" onClick={() => handleEdit(record)}>
              Chỉnh sửa
            </Button>
          </Space>
        );
      },
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
              <Upload {...uploadSimpleProps} disabled={importingSimple || loading}>
                <Button 
                  icon={<UploadOutlined />} 
                  loading={importingSimple}
                  disabled={loading}
                >
                  Import Excel
                </Button>
              </Upload>
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
          <Space>
            <Upload {...uploadProps} disabled={importing}>
              <Button icon={<UploadOutlined />} loading={importing}>
                Import Excel
              </Button>
            </Upload>
            <Button
              icon={<ReloadOutlined />}
              onClick={fetchFeedbacks}
              loading={loading}
            >
              Làm mới
            </Button>
          </Space>
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

      <Modal
        title="Chỉnh sửa Feedback"
        open={editModalVisible}
        onCancel={handleEditCancel}
        onOk={handleEditSubmit}
        confirmLoading={updatingFeedback}
        okText="Lưu"
        cancelText="Hủy"
        destroyOnClose
      >
        <Form form={editForm} layout="vertical">
          <Form.Item
            label="Sentiment"
            name="sentiment_label"
            rules={[{ required: true, message: 'Vui lòng chọn sentiment' }]}
          >
            <Select placeholder="Chọn sentiment">
              <Option value="POSITIVE">Tích cực</Option>
              <Option value="NEGATIVE">Tiêu cực</Option>
              <Option value="EXTREMELY_NEGATIVE">Rất tiêu cực</Option>
              <Option value="NEUTRAL">Trung tính</Option>
            </Select>
          </Form.Item>

          <Divider orientation="left" plain style={{ margin: '8px 0 16px' }}>
            Intent (Level 1 → Level 2 → Level 3)
          </Divider>

          <Form.Item label="Level 1" name="level1_id">
            <Select
              allowClear
              placeholder="Chọn Level 1"
              onChange={handleLevel1Change}
              loading={labelLoading && labelTree.length === 0}
            >
              {labelTree.map((level1) => (
                <Option key={level1.id} value={level1.id}>
                  {level1.name}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item label="Level 2" name="level2_id">
            <Select
              allowClear
              placeholder={selectedLevel1 ? 'Chọn Level 2' : 'Chọn Level 1 trước'}
              onChange={handleLevel2Change}
              disabled={!selectedLevel1}
            >
              {availableLevel2.map((level2) => (
                <Option key={level2.id} value={level2.id}>
                  {level2.name}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item label="Level 3" name="level3_id">
            <Select
              allowClear
              placeholder={selectedLevel2 ? 'Chọn Level 3' : 'Chọn Level 2 trước'}
              onChange={handleLevel3Change}
              disabled={!selectedLevel2}
            >
              {availableLevel3.map((level3) => (
                <Option key={level3.id} value={level3.id}>
                  {level3.name}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Text type="secondary" style={{ fontSize: '12px' }}>
            Có thể bỏ trống intent nếu muốn gỡ gán khỏi feedback này.
          </Text>
        </Form>
      </Modal>
    </div>
  );
};

export default FeedbackSentiment;

