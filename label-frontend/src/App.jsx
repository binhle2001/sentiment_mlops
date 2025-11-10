import React, { useState } from 'react';
import { ConfigProvider, Tabs, Layout } from 'antd';
import { TagsOutlined, CommentOutlined } from '@ant-design/icons';
import LabelManager from './components/LabelManager';
import FeedbackSentiment from './components/FeedbackSentiment';

const { Header, Content } = Layout;

function App() {
  const [activeTab, setActiveTab] = useState('labels');

  const items = [
    {
      key: 'labels',
      label: (
        <span>
          <TagsOutlined />
          Quản lý Nhãn
        </span>
      ),
      children: <LabelManager />,
    },
    {
      key: 'feedback',
      label: (
        <span>
          <CommentOutlined />
          Phân tích Sentiment
        </span>
      ),
      children: <FeedbackSentiment />,
    },
  ];

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
        },
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ background: '#fff', padding: '0 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
          <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 600, color: '#1890ff' }}>
            CXM BIDV MLOps - Label & Sentiment Management
          </h1>
        </Header>
        <Content style={{ backgroundColor: '#f0f2f5' }}>
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={items}
            style={{ padding: '0 24px' }}
            size="large"
          />
        </Content>
      </Layout>
    </ConfigProvider>
  );
}

export default App;



