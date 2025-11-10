import React from 'react';
import { ConfigProvider } from 'antd';
import LabelManager from './components/LabelManager';

function App() {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
        },
      }}
    >
      <div style={{ minHeight: '100vh', backgroundColor: '#f0f2f5' }}>
        <LabelManager />
      </div>
    </ConfigProvider>
  );
}

export default App;


