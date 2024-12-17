import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    chief_complaint: "",
    spo2: "",
    pulse_rate: "",
    systolic_bp: "",
  });

  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [explanation, setExplanation] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResponse(null);
    setExplanation("");

    try {
      const apiResponse = await axios.post("http://127.0.0.1:9000/main/triage", {
        chief_complaint: formData.chief_complaint,
        spo2: parseFloat(formData.spo2),
        pulse_rate: parseFloat(formData.pulse_rate),
        systolic_bp: parseFloat(formData.systolic_bp),
      });

      const data = apiResponse.data;
      setResponse(data);

      // Generate explanation based on positive impacts
      const positiveImpacts = [];
      if (data.pulse_rate_importance_impact === "positive")
        positiveImpacts.push("Pulse Rate");
      if (data.spo2_importance_impact === "positive")
        positiveImpacts.push("SpO₂ Level");
      if (data.systolic_bp_importance_impact === "positive")
        positiveImpacts.push("Systolic BP");

      const impactFeatures = positiveImpacts.join(", ") || "other factors";
      setExplanation(
        `This prediction is based on ${impactFeatures}. With the confidence of ${data.confidence_score_level.toFixed(
          2
        )}%, the predicted triage level is ${data.triage_level}.`
      );
    } catch (err) {
      console.error("Error:", err);
      setError("Failed to fetch triage level. Please try again.");
    }
  };

  return (
    <div className="container">
      <h1 className="title">TriML</h1>
      <h3>Predictive Triage Assistance Tool</h3>
      <form onSubmit={handleSubmit} className="form">
        <label>
          <strong>Chief Complaint</strong>
          <input
            type="text"
            name="chief_complaint"
            placeholder="e.g., Severe shortness of breath"
            value={formData.chief_complaint}
            onChange={handleChange}
            required
          />
        </label>
        <label>
          <strong>SpO₂ Level (%)</strong>
          <input
            type="number"
            name="spo2"
            placeholder="e.g., 92"
            value={formData.spo2}
            onChange={handleChange}
            required
          />
        </label>
        <label>
          <strong>Pulse Rate (bpm)</strong>
          <input
            type="number"
            name="pulse_rate"
            placeholder="e.g., 108"
            value={formData.pulse_rate}
            onChange={handleChange}
            required
          />
        </label>
        <label>
          <strong>Systolic Blood Pressure (mmHg)</strong>
          <input
            type="number"
            name="systolic_bp"
            placeholder="e.g., 120"
            value={formData.systolic_bp}
            onChange={handleChange}
            required
          />
        </label>
        <button type="submit" className="submit-button">
          Get Prediction
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {response && (
        <div className="result">
          <h3 className="triage-level">Predicted Triage Level: {response.triage_level}</h3>
          <div className="left-aligned-text">
            <p>{explanation}</p>
            <p>For more information, see below:</p>
            <div className="description">
              <li>Confidence Score (Level): {response.confidence_score_level}</li>
              <li>Confidence Score (Range): {response.confidence_score_range}</li>
              <li>Pulse Rate Impact: {response.pulse_rate_importance_impact}</li>
              <li>SpO₂ Impact: {response.spo2_importance_impact}</li>
              <li>Systolic BP Impact: {response.systolic_bp_importance_impact}</li>
              <li>Triage Range: {response.triage_range}</li>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
