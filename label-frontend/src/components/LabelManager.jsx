import React, { useState, useEffect } from 'react';
import {
  Typography,
  Button,
  Space,
  message,
  Spin,
  Card,
  Statistic,
  Row,
  Col,
  Input,
  Upload,
} from 'antd';
import {
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
  AppstoreAddOutlined,
  UploadOutlined,
  DownloadOutlined,
} from '@ant-design/icons';
import LabelTree from './LabelTree';
import LabelForm from './LabelForm';
import BulkLabelForm from './BulkLabelForm';
import { labelAPI } from '../services/api';

const { Text } = Typography;

const LabelManager = () => {
  const [labels, setLabels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [formVisible, setFormVisible] = useState(false);
  const [bulkFormVisible, setBulkFormVisible] = useState(false);
  const [editingLabel, setEditingLabel] = useState(null);
  const [parentLabel, setParentLabel] = useState(null);
  const [stats, setStats] = useState({ level1: 0, level2: 0, level3: 0, total: 0 });
  const [searchText, setSearchText] = useState('');
  const [syncing, setSyncing] = useState(false);

  useEffect(() => {
    loadLabels();
    loadStats();
  }, []);

  const loadLabels = async () => {
    setLoading(true);
    try {
      const tree = await labelAPI.getTree();
      setLabels(tree);
    } catch (error) {
      message.error('Failed to load labels');
      console.error('Error loading labels:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const [level1, level2, level3, total] = await Promise.all([
        labelAPI.getLabels({ level: 1 }),
        labelAPI.getLabels({ level: 2 }),
        labelAPI.getLabels({ level: 3 }),
        labelAPI.getLabels({}),
      ]);
      setStats({
        level1: level1.total,
        level2: level2.total,
        level3: level3.total,
        total: total.total,
      });
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const handleCreate = () => {
    setEditingLabel(null);
    setParentLabel(null);
    setFormVisible(true);
  };

  const handleBulkCreate = () => {
    setBulkFormVisible(true);
  };

  const handleEdit = (label) => {
    setEditingLabel(label);
    setParentLabel(null);
    setFormVisible(true);
  };

  const handleAddChild = (parentLabel) => {
    setEditingLabel(null);
    setParentLabel(parentLabel);
    setBulkFormVisible(true);  // Open bulk form instead of single form
  };

  const handleDelete = async (labelId) => {
    try {
      await labelAPI.deleteLabel(labelId);
      message.success('Label deleted successfully');
      loadLabels();
      loadStats();
    } catch (error) {
      message.error(error.response?.data?.detail || 'Failed to delete label');
    }
  };

  const handleFormSuccess = () => {
    loadLabels();
    loadStats();
  };

  const handleRefresh = () => {
    loadLabels();
    loadStats();
  };

  const handleSyncLabels = async (file) => {
    if (!file) {
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setSyncing(true);
      const result = await labelAPI.syncLabels(formData);
      message.success(
        `Đã sync ${result.total} labels: ${result.created} mới, ${result.updated} cập nhật, ${result.unchanged} không thay đổi`
      );
      if (result.impacted_feedbacks > 0) {
        message.info(
          `Có ${result.impacted_feedbacks} feedback bị ảnh hưởng và đã được xử lý lại`
        );
      }
      await loadLabels();
      await loadStats();
    } catch (error) {
      const detail = error?.response?.data?.detail || 'Không thể sync labels';
      message.error(detail);
      console.error('Error syncing labels:', error);
    } finally {
      setSyncing(false);
    }
  };

  const handleExportBackup = async () => {
    try {
      const blob = await labelAPI.exportLabelsBackup();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `labels_backup_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      message.success('Đã export labels backup thành công');
    } catch (error) {
      const detail = error?.response?.data?.detail || 'Không thể export backup';
      message.error(detail);
      console.error('Error exporting backup:', error);
    }
  };

  const uploadProps = {
    accept: '.json',
    showUploadList: false,
    beforeUpload: (file) => {
      handleSyncLabels(file);
      return false;
    },
  };

  // Filter labels based on search
  const filterLabels = (labels, searchText) => {
    if (!searchText) return labels;
    
    return labels
      .map((label) => {
        const matchesSearch = label.name.toLowerCase().includes(searchText.toLowerCase());
        const filteredChildren = label.children ? filterLabels(label.children, searchText) : [];
        
        if (matchesSearch || filteredChildren.length > 0) {
          return {
            ...label,
            children: filteredChildren,
          };
        }
        return null;
      })
      .filter(Boolean);
  };

  const filteredLabels = filterLabels(labels, searchText);

  return (
    <div style={{ padding: '24px', backgroundColor: '#f0f2f5', minHeight: '100vh' }}>
      {/* Action Buttons */}
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'flex-end' }}>
        <Space>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={handleCreate}
          >
            Create Label
          </Button>
          <Button
            type="primary"
            icon={<AppstoreAddOutlined />}
            onClick={handleBulkCreate}
            style={{ background: '#52c41a', borderColor: '#52c41a' }}
          >
            Bulk Create
          </Button>
          <Upload {...uploadProps} disabled={syncing}>
            <Button 
              icon={<UploadOutlined />} 
              loading={syncing}
              disabled={syncing}
            >
              Sync Labels
            </Button>
          </Upload>
          <Button 
            icon={<DownloadOutlined />} 
            onClick={handleExportBackup}
          >
            Export Backup
          </Button>
          <Button icon={<ReloadOutlined />} onClick={handleRefresh}>
            Refresh
          </Button>
        </Space>
      </div>
        {/* Statistics */}
        <Row gutter={16} style={{ marginBottom: '24px' }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="Total Labels"
                value={stats.total}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Level 1"
                value={stats.level1}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Level 2"
                value={stats.level2}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Level 3"
                value={stats.level3}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
        </Row>

        {/* Search */}
        <Card style={{ marginBottom: '24px' }}>
          <Input
            placeholder="Search labels..."
            prefix={<SearchOutlined />}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            allowClear
            size="large"
          />
        </Card>

        {/* Label Tree */}
        <Card
          title={
            <Space>
              <Text strong>Label Hierarchy</Text>
              <Text type="secondary">
                ({filteredLabels.length} root labels)
              </Text>
            </Space>
          }
        >
          {loading ? (
            <div style={{ textAlign: 'center', padding: '50px' }}>
              <Spin size="large" />
            </div>
          ) : filteredLabels.length > 0 ? (
            <LabelTree
              data={filteredLabels}
              onEdit={handleEdit}
              onDelete={handleDelete}
              onAddChild={handleAddChild}
              loading={loading}
            />
          ) : (
            <div style={{ textAlign: 'center', padding: '50px' }}>
              <Text type="secondary">
                {searchText ? 'No labels found matching your search' : 'No labels yet. Create your first label!'}
              </Text>
            </div>
          )}
        </Card>

      {/* Form Modal */}
      <LabelForm
        visible={formVisible}
        onClose={() => setFormVisible(false)}
        onSuccess={handleFormSuccess}
        editingLabel={editingLabel}
        parentLabel={parentLabel}
      />

      {/* Bulk Create Form Modal */}
      <BulkLabelForm
        visible={bulkFormVisible}
        onClose={() => {
          setBulkFormVisible(false);
          setParentLabel(null);
        }}
        onSuccess={handleFormSuccess}
        parentLabel={parentLabel}
      />
    </div>
  );
};

export default LabelManager;



