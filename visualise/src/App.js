import React, { useEffect, useState } from 'react';
import Graph from './components/Graph';

const fullscreenStyle = {
  position: 'relative',
  height: '100vh',
  width: '100vw',
};

function App() {

  return (
          <div style={fullscreenStyle}>
              <Graph />
          </div>
  );
}

export default App;
