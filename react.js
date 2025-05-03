import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("image", file);
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData);
      setResult(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <h1>Diabetic Retinopathy Classifier</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      {result && (
        <div>
          <p>Predicted Class: {result.class}</p>
          <p>Confidence: {result.confidence.toFixed(2)}%</p>
          <img src={result.heatmap_url} alt="Grad-CAM++ Heatmap" style={{ width: '300px' }} />
        </div>
      )}
    </div>
  );
}

export default App;
