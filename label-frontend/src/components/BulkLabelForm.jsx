import React, { useState, useEffect } from 'react';
import { Modal, Form, Input, Select, Button, Space, message, Alert, Table, Tag } from 'antd';
import { PlusOutlined, MinusCircleOutlined } from '@ant-design/icons';
import { labelAPI } from '../services/api';

const { TextArea } = Input;
const { Option } = Select;

const BulkLabelForm = ({ visible, onClose, onSuccess }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [level1Labels, setLevel1Labels] = useState([]);
  const [level2LabelsCache, setLevel2LabelsCache] = useState({});
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (visible) {
      loadLevel1Labels();
      form.resetFields();
      setResult(null);
    }
  }, [visible]);

  const loadLevel1Labels = async () => {
    try {
      const response = await labelAPI.getLabels({ level: 1 });
      setLevel1Labels(response.labels || []);
    } catch (error) {
      console.error('Failed to load level 1 labels:', error);
    }
  };

  const loadLevel2LabelsForParent = async (parentId) => {
    if (level2LabelsCache[parentId]) {
      return level2LabelsCache[parentId];
    }

    try {
      const response = await labelAPI.getChildren(parentId);
      setLevel2LabelsCache(prev => ({ ...prev, [parentId]: response || [] }));
      return response || [];
    } catch (error) {
      console.error('Failed to load level 2 labels:', error);
      return [];
    }
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);

      // Prepare labels array
      const labels = values.labels.map(label => ({
        name: label.name,
        level: label.level,
        parent_id: label.parent_id || null,
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

  return (
    <Modal
      title="Create Multiple Labels"
      open={visible}
      onOk={handleSubmit}
      onCancel={handleCancel}
      confirmLoading={loading}
      width={900}
      okText="Create All"
    >
      {result && (
        <Alert
          message={`Total: ${result.total} | Success: ${result.successful} | Failed: ${result.failed}`}
          type={result.failed === 0 ? 'success' : 'warning'}
          style={{ marginBottom: 16 }}
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
          initialValues={{ labels: [{ level: 1 }] }}
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
                      <div style={{ position: 'absolute', top: 10, right: 10 }}>
                        <MinusCircleOutlined
                          onClick={() => remove(name)}
                          style={{ fontSize: '18px', color: '#ff4d4f', cursor: 'pointer' }}
                        />
                      </div>

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
                        <Input placeholder="Enter label name" />
                      </Form.Item>

                      <Form.Item
                        {...restField}
                        name={[name, 'level']}
                        label="Level"
                        rules={[{ required: true, message: 'Required' }]}
                      >
                        <Select placeholder="Select level">
                          <Option value={1}>Level 1 (Top)</Option>
                          <Option value={2}>Level 2 (Middle)</Option>
                          <Option value={3}>Level 3 (Bottom)</Option>
                        </Select>
                      </Form.Item>

                      <Form.Item noStyle shouldUpdate={(prevValues, currentValues) => {
                        return prevValues.labels?.[name]?.level !== currentValues.labels?.[name]?.level;
                      }}>
                        {({ getFieldValue }) => {
                          const level = getFieldValue(['labels', name, 'level']);
                          
                          if (level === 2) {
                            return (
                              <Form.Item
                                {...restField}
                                name={[name, 'parent_id']}
                                label="Parent (Level 1)"
                                rules={[{ required: true, message: 'Required' }]}
                              >
                                <Select placeholder="Select parent label">
                                  {level1Labels.map((label) => (
                                    <Option key={label.id} value={label.id}>
                                      {label.name}
                                    </Option>
                                  ))}
                                </Select>
                              </Form.Item>
                            );
                          }

                          if (level === 3) {
                            return (
                              <>
                                <Form.Item
                                  {...restField}
                                  name={[name, 'level1_parent']}
                                  label="Parent Level 1"
                                  rules={[{ required: true, message: 'Required' }]}
                                >
                                  <Select
                                    placeholder="Select level 1 label"
                                    onChange={(value) => {
                                      loadLevel2LabelsForParent(value);
                                      form.setFieldValue(['labels', name, 'parent_id'], undefined);
                                    }}
                                  >
                                    {level1Labels.map((label) => (
                                      <Option key={label.id} value={label.id}>
                                        {label.name}
                                      </Option>
                                    ))}
                                  </Select>
                                </Form.Item>

                                <Form.Item noStyle shouldUpdate={(prevValues, currentValues) => {
                                  return prevValues.labels?.[name]?.level1_parent !== currentValues.labels?.[name]?.level1_parent;
                                }}>
                                  {({ getFieldValue }) => {
                                    const level1Parent = getFieldValue(['labels', name, 'level1_parent']);
                                    const level2Options = level2LabelsCache[level1Parent] || [];

                                    return (
                                      <Form.Item
                                        {...restField}
                                        name={[name, 'parent_id']}
                                        label="Parent (Level 2)"
                                        rules={[{ required: true, message: 'Required' }]}
                                      >
                                        <Select
                                          placeholder="Select level 2 label"
                                          disabled={!level1Parent}
                                        >
                                          {level2Options.map((label) => (
                                            <Option key={label.id} value={label.id}>
                                              {label.name}
                                            </Option>
                                          ))}
                                        </Select>
                                      </Form.Item>
                                    );
                                  }}
                                </Form.Item>
                              </>
                            );
                          }

                          return null;
                        }}
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
                    onClick={() => add({ level: 1 })}
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

