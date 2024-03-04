import React, { useState } from 'react';
import { TextField, Button, Grid, Typography, Paper, useTheme } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import ClearIcon from '@mui/icons-material/Clear';

function ParameterPanel({ setParameters, stopSimulation, clearGraph }) {
  const theme = useTheme();
  const [rho, setRho] = useState(1);
  const [nu, setNu] = useState(2);
  const [numIters, setNumIters] = useState(10);


  const handleRhoChange = (event) => setRho(event.target.value);
  const handleNuChange = (event) => setNu(event.target.value);
  const handleNumItersChange = (event) => setNumIters(event.target.value);

  const handleSubmit = (event) => {
    event.preventDefault();
    setParameters({ rho, nu, numIters });
  };

  const handleStopClick = (event) => {
    event.preventDefault();
    stopSimulation();
  };

  const handleClearClick = (event) => {
    event.preventDefault();
    clearGraph();
  };

  return (
    <div style={{ paddingBottom: '16px' }}>
      <Paper elevation={3} style={{ maxWidth: '300px', padding: '16px', backgroundColor: theme.palette.background.paper }}>
        <Typography variant="h6" style={{ marginBottom: '12px', textAlign: 'left' }}>🌐 Generate Community</Typography>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={1} alignItems="center">
            
            {/* Fields */}
            <Grid item xs={4}>
              <TextField fullWidth label="Reinforcement" type="number" value={rho} onChange={handleRhoChange} variant="outlined" size="small" />
            </Grid>
            <Grid item xs={4}>
              <TextField fullWidth label="Novelty" type="number" value={nu} onChange={handleNuChange} variant="outlined" size="small" />
            </Grid>
            <Grid item xs={4}>
              <TextField fullWidth label="Iterations" type="number" value={numIters} onChange={handleNumItersChange} variant="outlined" size="small" />
            </Grid>

            {/* Buttons */}
            <Grid item xs={12} style={{ textAlign: 'center', marginTop: '8px' }}>
            <Button type="submit" variant="contained" color="primary" style={{ padding: '6px 15px', marginRight: '10px' }}>
              <PlayArrowIcon />
            </Button>
            <Button variant="contained" color="secondary" style={{ padding: '6px 15px', marginRight: '10px' }} onClick={handleStopClick}>
              <StopIcon />
            </Button>
            <Button variant="contained" color="secondary" style={{ padding: '6px 15px', marginRight: '10px' }} onClick={handleClearClick}>
              <ClearIcon />
            </Button>
          </Grid>
          </Grid>
        </form>
      </Paper>
    </div>
  );
}

export default ParameterPanel;
