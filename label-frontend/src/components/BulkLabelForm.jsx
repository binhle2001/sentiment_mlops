import React, { useState, useEffect } from 'react';
import { Modal, Form, Input, Select, Button, Space, message, Alert, Table, Tag } from 'antd';
import { PlusOutlined, MinusCircleOutlined } from '@ant-design/icons';
import { labelAPI } from '../services/api';

const { TextArea } = Input;
const { Option } = Select;

const BulkLabelForm = ({ visible, onClose, onSuccess, parentLabel }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (visible) {
      form.resetFields();
      setResult(null);
      // Initialize with one empty label
      form.setFieldsValue({ labels: [{ name: '', description: '' }] });
    }
  }, [visible, parentLabel]);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);

      // Calculate level for children
      const childLevel = parentLabel ? parentLabel.level + 1 : 1;
      const parent_id = parentLabel ? parentLabel.id : null;

      // Prepare labels array - all with same level and parent
      const labels = values.labels.map(label => ({
        name: label.name,
        level: childLevel,
        parent_id: parent_id,
        description: label.description || null,
      }));

      const response = await labelAPI.createLabelsBulk(labels);
      
      if (response.successful > 0) {
        message.success(`Successfully created ${response.successful} label(s)`);
      }
      
      if (response.failed > 0) {
        message.warning(`${response.failed} label(s) failed to create`);
      }

      setResult(response);
      
      // If all successful, close modal
      if (response.failed === 0) {
        setTimeout(() => {
          form.resetFields();
          setResult(null);
          onSuccess();
          onClose();
        }, 1500);
      }
    } catch (error) {
      if (error.response) {
        message.error(error.response.data?.detail || 'Operation failed');
      } else if (error.errorFields) {
        message.error('Please fill in all required fields');
      } else {
        message.error('Operation failed');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    setResult(null);
    onClose();
  };

  const resultColumns = [
    {
      title: 'Index',
      dataIndex: 'index',
      key: 'index',
      width: 70,
      render: (index) => index + 1,
    },
    {
      title: 'Status',
      dataIndex: 'success',
      key: 'success',
      width: 100,
      render: (success) => (
        <Tag color={success ? 'success' : 'error'}>
          {success ? 'Success' : 'Failed'}
        </Tag>
      ),
    },
    {
      title: 'Label Name',
      dataIndex: 'label',
      key: 'name',
      render: (label) => label?.name || '-',
    },
    {
      title: 'Message',
      dataIndex: 'error',
      key: 'error',
      render: (error, record) => error || (record.success ? 'Created successfully' : '-'),
    },
  ];

  const getModalTitle = () => {
    if (parentLabel) {
      return `Create Multiple Children for "${parentLabel.name}"`;
    }
    return 'Create Multiple Level 1 Labels';
  };

  return (
    <Modal
      title={getModalTitle()}
      open={visible}
      onOk={handleSubmit}
      onCancel={handleCancel}
      confirmLoading={loading}
      width={800}
      okText="Create All"
    >
      {result && (
        <Alert
          message={`Total: ${result.total} | Success: ${result.successful} | Failed: ${result.failed}`}
          type={result.failed === 0 ? 'success' : 'warning'}
          style={{ marginBottom: 16 }}
        />
      )}

      {parentLabel && (
        <Alert
          message={`Creating Level ${parentLabel.level + 1} labels under "${parentLabel.name}"`}
          type="info"
          style={{ marginBottom: 16 }}
          showIcon
        />
      )}

      {result ? (
        <Table
          columns={resultColumns}
          dataSource={result.results}
          rowKey="index"
          pagination={false}
          size="small"
          scroll={{ y: 400 }}
        />
      ) : (
        <Form
          form={form}
          layout="vertical"
          initialValues={{ labels: [{ name: '', description: '' }] }}
        >
          <Form.List name="labels">
            {(fields, { add, remove }) => (
              <>
                <div style={{ maxHeight: '500px', overflowY: 'auto', paddingRight: '10px' }}>
                  {fields.map(({ key, name, ...restField }, index) => (
                    <div
                      key={key}
                      style={{
                        border: '1px solid #d9d9d9',
                        borderRadius: '8px',
                        padding: '16px',
                        marginBottom: '16px',
                        position: 'relative',
                        backgroundColor: '#fafafa',
                      }}
                    >
                      {fields.length > 1 && (
                        <div style={{ position: 'absolute', top: 10, right: 10 }}>
                          <MinusCircleOutlined
                            onClick={() => remove(name)}
                            style={{ fontSize: '18px', color: '#ff4d4f', cursor: 'pointer' }}
                          />
                        </div>
                      )}

                      <div style={{ fontWeight: 'bold', marginBottom: '12px', color: '#1890ff' }}>
                        Label #{index + 1}
                      </div>

                      <Form.Item
                        {...restField}
                        name={[name, 'name']}
                        label="Label Name"
                        rules={[
                          { required: true, message: 'Required' },
                          { max: 255, message: 'Max 255 characters' },
                        ]}
                      >
                        <Input placeholder="Enter label name" autoFocus={index === 0} />
                      </Form.Item>

                      <Form.Item
                        {...restField}
                        name={[name, 'description']}
                        label="Description"
                      >
                        <TextArea rows={2} placeholder="Optional description" />
                      </Form.Item>
                    </div>
                  ))}
                </div>

                <Form.Item>
                  <Button
                    type="dashed"
                    onClick={() => add({ name: '', description: '' })}
                    block
                    icon={<PlusOutlined />}
                  >
                    Add Another Label
                  </Button>
                </Form.Item>
              </>
            )}
          </Form.List>
        </Form>
      )}
    </Modal>
  );
};

export default BulkLabelForm;

