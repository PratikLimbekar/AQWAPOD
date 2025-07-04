import React, { useState } from "react";
import waterQualityData from "./water_quality.json";
import './location.css';

const Location = () => {
  const locations = [
"GODAVARI AT JAYAKWADI DAM, AURNAGABAD,MAHARASHTRA",
"GODAVARI RIVER NEAR SOMESHWAR TEMPLE.",
"GODAVARI RIVER AT SAIKHEDA.",
"GODAVARI RIVER AT HANUMAN GHAT, NASHIK CITY.",
"GODAVARI RIVER AT NANDUR- MADMESHWAR DAM.",
"GODAVARI RIVER AT KAPILA- GODAVARI CONFLUENCE POINT, TAPOVAN.",
"GODAVARI RIVER NEAR TAPOVAN.",
"GODAVARI AT PANCHAVATI AT RAMKUND,MAHARASHTRA",
"GODAVARI AT NASIK D/S, MAHARASHTRA",
"GODAVARI AT U/S OF GANGAPUR DAM,NASIK,MAHARASHTRA",
"GODAVARI RIVER AT U/S OF AURANGABAD RESERVOIR, KAIGAON TOKKA NEAR KAIGAON BRIDGE.",
"GODAVARI RIVER AT U/S OF PAITHAN AT PAITHAN INTAKE PUMP HOUSE AT JAYAKWADI ",
"GODAVARI RIVER AT D/S OF PAITHAN AT PATHEGAON BRIDGE.",
"GODAVARI RIVER AT JALNA INTAKE WATER PUMP HOUSE, SHAHABAD.",
"GODAVARI AT DHALEGAON, MAHARASHTRA",
"GODAVARI AT NANDED, MAHARASHTRA",
"GODAVARI AT RAHER, MAHARASHTRA",
"GODAVARI RIVER AT LATUR WATER INTAKE NEAR PUMP HOUSE AT DHAMEGAON.",
"GODAVARI AT BASARA, ADILABAD",
"GODAVARI  AT MANCHERIAL, NEAR RLY BDG B/C OF RALLAVAGU",
"GODAVARI AT RAMAGUNDAM D/S, NEAR FCI INTAKE WELL, KARIMNAGAR",
"GODAVARI AT GODAVARIKHANI, NEAR BATHING GHAT, KARIMNAGAR",
"GODAVARI AT RAMAGUNDAM U/S , KARIMNAGAR",
"GODAVARI,  D/S OF RAMANUGUNDAM,",
"GODAVARI AT KAMALPUR U/S M/S AP RAYONS LTD. INTAKE WELL, WARANGAL",
"GODAVARI AT KAMALPUR D/S AT M/S. AP RAYONS LTD. DISCHARGE POINT, WARANGAL",
"GODAVARI AT MANCHERIAL, A.P.",
"GODAVARI AT BHADRACHALAM U/S BATHING GHAT, KHAMMAM",
"GODAVARI AT BHADRACHALAM D/S BATHING GHAT, KHAMMAM",
"GODAVARI AT BURGAMPAHAD, KHAMMAM",
"GODAVARI AT RAJAMUNDRY U/S OF NALLA CHANNEL",
"GODAVARI AT POLAVARAM, A.P.",
"GODAVARI AT RAJAHMUNDRY U/S, A.P.",
"GODAVARI AT RAJAMUNDRY D/S OF NALLA CHANNEL",
"GODAVARI AT RAJAHMUNDRY D/S, A.P.",

  ];

  const [selectedLocation, setSelectedLocation] = useState("");
  const [prediction, setPrediction] = useState(null);

  const selectedData = waterQualityData.find(
    (entry) => entry.LOCATIONS.trim() === selectedLocation.trim()
  );

  const handlePredict = async () => {
    if (!selectedData) return;
    
    const requestData = {
      pH: selectedData["pH  : Mean  : 6.5-8.5"],
      BOD: selectedData["B.O.D. (mg/l)  : Mean  : < 3 mg/l"],
      DO: selectedData["D.O. (mg/l)  : Mean  : > 4 mg/l"],
      Nitrate: selectedData["NITRATE- N+ NITRITE-N (mg/l)  : Mean"],
      Temperature_Mean: selectedData["TEMPERATURE ºC  : Mean"],
      Conductivity_Mean: selectedData["CONDUCTIVITY (µmhos/cm)  : Mean"],
      Fecal_Coliform: selectedData["FECAL COLIFORM (MPN/100ml)  : Mean  : < 2500 MPN/100ml"],
      Total_Coliform: selectedData["TOTAL COLIFORM (MPN/100ml)  : Mean  : < 5000 MPN/100ml"]
    };
    
    const response = await fetch("http://localhost:5000/location", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestData),
    });
    
    const result = await response.json();
    setPrediction(result.potability);
  };

  return (
    <div>
      <h1>Location Based Prediction: Godavari River</h1>
      <p>Please select the location from the dropdown below: </p>
      <select
        className='dropdown'
        value={selectedLocation}
        onChange={(e) => setSelectedLocation(e.target.value)}
      >
        <option value="">Select a location</option>
        {locations.map((location, index) => (
          <option key={index} value={location}>{location}</option>
        ))}
      </select>
      <p>Selected Location: {selectedLocation}</p>
      
      {selectedData && (
        <div>
          <h2>Water Quality Parameters</h2>
          <p><strong>pH:</strong> {selectedData["pH  : Mean  : 6.5-8.5"]}</p>
          <p><strong>BOD (mg/l):</strong> {selectedData["B.O.D. (mg/l)  : Mean  : < 3 mg/l"]}</p>
          <p><strong>DO (mg/l):</strong> {selectedData["D.O. (mg/l)  : Mean  : > 4 mg/l"]}</p>
          <p><strong>Nitrate (mg/l):</strong> {selectedData["NITRATE- N+ NITRITE-N (mg/l)  : Mean"]}</p>
          <p><strong>Mean Temperature:</strong> {selectedData["TEMPERATURE ºC  : Mean"]}</p>
          <p><strong>Mean Conductivity:</strong> {selectedData["CONDUCTIVITY (µmhos/cm)  : Mean"]}</p>
          <p><strong>Mean Fecal Coliform:</strong> {selectedData["FECAL COLIFORM (MPN/100ml)  : Mean  : < 2500 MPN/100ml"]}</p>          
          <p><strong>Mean Total Coliform:</strong> {selectedData["TOTAL COLIFORM (MPN/100ml)  : Mean  : < 5000 MPN/100ml"]}</p>          
          <button 
            className="predict" 
            onClick={handlePredict}
          >
            Predict Potability
          </button>
          {prediction !== null && (
            <p>Predicted Potability: {prediction}</p>
          )}
        </div>
      )}
    </div>
  );
};

export default Location;
