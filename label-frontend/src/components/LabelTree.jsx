import React from 'react';
import { Tree, Space, Button, Popconfirm, Typography, Tag } from 'antd';
import {
  EditOutlined,
  DeleteOutlined,
  PlusOutlined,
  FolderOutlined,
  FileOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

const LabelTree = ({ data, onEdit, onDelete, onAddChild, loading }) => {
  // Convert label data to tree format for Ant Design Tree
  const convertToTreeData = (labels) => {
    return labels.map((label) => ({
      title: renderTreeNode(label),
      key: label.id,
      icon: getIcon(label.level),
      children: label.children && label.children.length > 0 
        ? convertToTreeData(label.children) 
        : undefined,
      data: label,
    }));
  };

  const getIcon = (level) => {
    if (level === 1) return <FolderOutlined style={{ color: '#1890ff' }} />;
    if (level === 2) return <FolderOutlined style={{ color: '#52c41a' }} />;
    return <FileOutlined style={{ color: '#faad14' }} />;
  };

  const getLevelTag = (level) => {
    const colors = { 1: 'blue', 2: 'green', 3: 'orange' };
    return <Tag color={colors[level]}>L{level}</Tag>;
  };

  const renderTreeNode = (label) => {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '4px 8px',
          width: '100%',
        }}
      >
        <Space>
          {getLevelTag(label.level)}
          <Text strong>{label.name}</Text>
          {label.description && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              ({label.description})
            </Text>
          )}
        </Space>
        <Space size="small">
          {label.level < 3 && (
            <Button
              type="text"
              size="small"
              icon={<PlusOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                onAddChild(label);
              }}
              title="Add child"
            />
          )}
          <Button
            type="text"
            size="small"
            icon={<EditOutlined />}
            onClick={(e) => {
              e.stopPropagation();
              onEdit(label);
            }}
            title="Edit"
          />
          <Popconfirm
            title="Delete Label"
            description={
              label.children && label.children.length > 0
                ? 'This will also delete all child labels. Are you sure?'
                : 'Are you sure you want to delete this label?'
            }
            onConfirm={(e) => {
              e.stopPropagation();
              onDelete(label.id);
            }}
            onCancel={(e) => e.stopPropagation()}
            okText="Yes"
            cancelText="No"
          >
            <Button
              type="text"
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={(e) => e.stopPropagation()}
              title="Delete"
            />
          </Popconfirm>
        </Space>
      </div>
    );
  };

  const treeData = convertToTreeData(data);

  return (
    <Tree
      showIcon
      defaultExpandAll
      treeData={treeData}
      style={{ 
        backgroundColor: 'white', 
        padding: '16px',
        borderRadius: '8px',
      }}
      blockNode
    />
  );
};

export default LabelTree;







