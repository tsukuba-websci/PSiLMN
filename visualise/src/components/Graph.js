import React, { useEffect, useState } from 'react';
import { ForceGraph2D } from 'react-force-graph';

function GraphComponent() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });

  useEffect(() => {
    fetch('data/networks/fully_connected_network/10.json')
      .then(response => response.json())
      .then(data => {
        // Directly set the fetched data to graphData state
        setGraphData(data);
      })
      .catch(error => console.error("Fetching data failed", error));
  }, []);
  
  return (
    <ForceGraph2D
      graphData={graphData}
    />
  );
}

export default GraphComponent;
