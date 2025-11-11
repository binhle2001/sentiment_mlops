import React, { useState, useEffect } from 'react';
import { Modal, Form, Input, Select, message } from 'antd';
import { labelAPI } from '../services/api';

const { TextArea } = Input;
const { Option } = Select;

const LabelForm = ({ visible, onClose, onSuccess, editingLabel, parentLabel }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [level1Labels, setLevel1Labels] = useState([]);
  const [level2Labels, setLevel2Labels] = useState([]);
  const [selectedLevel, setSelectedLevel] = useState(1);
  const [selectedLevel1, setSelectedLevel1] = useState(null);

  useEffect(() => {
    if (visible) {
      loadLevel1Labels();
      
      // If editing, populate form
      if (editingLabel) {
        form.setFieldsValue({
          name: editingLabel.name,
          description: editingLabel.description,
        });
      } else {
        // If creating with a parent, set level and parent
        if (parentLabel) {
          const level = parentLabel.level + 1;
          setSelectedLevel(level);
          form.setFieldsValue({
            level: level,
            parent_id: parentLabel.id,
          });
          
          if (level === 3) {
            // Load level 1 and set it, then load level 2 options
            loadLevel2LabelsForParent(parentLabel.parent_id);
            setSelectedLevel1(parentLabel.parent_id);
          }
        } else {
          form.resetFields();
          setSelectedLevel(1);
        }
      }
    }
  }, [visible, editingLabel, parentLabel]);

  const loadLevel1Labels = async () => {
    try {
      const response = await labelAPI.getLabels({ level: 1 });
      setLevel1Labels(response.labels || []);
    } catch (error) {
      console.error('Failed to load level 1 labels:', error);
    }
  };

  const loadLevel2LabelsForParent = async (parentId) => {
    try {
      const response = await labelAPI.getChildren(parentId);
      setLevel2Labels(response || []);
    } catch (error) {
      console.error('Failed to load level 2 labels:', error);
    }
  };

  const handleLevelChange = (level) => {
    setSelectedLevel(level);
    form.setFieldsValue({ parent_id: undefined });
    setSelectedLevel1(null);
    setLevel2Labels([]);
  };

  const handleLevel1Change = (level1Id) => {
    setSelectedLevel1(level1Id);
    form.setFieldsValue({ parent_id: undefined });
    if (selectedLevel === 3) {
      loadLevel2LabelsForParent(level1Id);
    }
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);

      if (editingLabel) {
        // Update existing label (only name and description)
        await labelAPI.updateLabel(editingLabel.id, {
          name: values.name,
          description: values.description,
        });
        message.success('Label updated successfully');
      } else {
        // Create new label
        const createData = {
          name: values.name,
          level: values.level,
          parent_id: values.parent_id || null,
          description: values.description || null,
        };
        await labelAPI.createLabel(createData);
        message.success('Label created successfully');
      }

      form.resetFields();
      setSelectedLevel(1);
      setSelectedLevel1(null);
      setLevel2Labels([]);
      onSuccess();
      onClose();
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
    setSelectedLevel(1);
    setSelectedLevel1(null);
    setLevel2Labels([]);
    onClose();
  };

  return (
    <Modal
      title={editingLabel ? 'Edit Label' : 'Create New Label'}
      open={visible}
      onOk={handleSubmit}
      onCancel={handleCancel}
      confirmLoading={loading}
      width={600}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{ level: 1 }}
      >
        <Form.Item
          name="name"
          label="Label Name"
          rules={[
            { required: true, message: 'Please enter label name' },
            { max: 255, message: 'Name must be less than 255 characters' },
          ]}
        >
          <Input placeholder="Enter label name" />
        </Form.Item>

        {!editingLabel && (
          <>
            <Form.Item
              name="level"
              label="Level"
              rules={[{ required: true, message: 'Please select level' }]}
            >
              <Select
                placeholder="Select level"
                onChange={handleLevelChange}
                disabled={!!parentLabel}
              >
                <Option value={1}>Level 1 (Top)</Option>
                <Option value={2}>Level 2 (Middle)</Option>
                <Option value={3}>Level 3 (Bottom)</Option>
              </Select>
            </Form.Item>

            {selectedLevel === 2 && (
              <Form.Item
                name="parent_id"
                label="Parent (Level 1)"
                rules={[{ required: true, message: 'Please select parent label' }]}
              >
                <Select placeholder="Select parent label">
                  {level1Labels.map((label) => (
                    <Option key={label.id} value={label.id}>
                      {label.name}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            )}

            {selectedLevel === 3 && (
              <>
                <Form.Item
                  label="Parent Level 1"
                  rules={[{ required: true, message: 'Please select level 1 label' }]}
                >
                  <Select
                    placeholder="Select level 1 label"
                    onChange={handleLevel1Change}
                    value={selectedLevel1}
                  >
                    {level1Labels.map((label) => (
                      <Option key={label.id} value={label.id}>
                        {label.name}
                      </Option>
                    ))}
                  </Select>
                </Form.Item>

                <Form.Item
                  name="parent_id"
                  label="Parent (Level 2)"
                  rules={[{ required: true, message: 'Please select parent label' }]}
                >
                  <Select
                    placeholder="Select level 2 label"
                    disabled={!selectedLevel1}
                  >
                    {level2Labels.map((label) => (
                      <Option key={label.id} value={label.id}>
                        {label.name}
                      </Option>
                    ))}
                  </Select>
                </Form.Item>
              </>
            )}
          </>
        )}

        <Form.Item
          name="description"
          label="Description"
        >
          <TextArea
            rows={4}
            placeholder="Enter description (optional)"
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default LabelForm;




