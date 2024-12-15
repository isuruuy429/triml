from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Triage Range API endpoint
TRIAGE_RANGE_API = "http://127.0.0.1:8000/predict/triage/range"

@app.route('/main/triage', methods=['POST'])
def main_triage():
    # Step 1: Get user input
    user_input = request.get_json()

    # Validate required fields
    required_fields = ['chief_complaint', 'systolic_bp', 'spo2', 'pulse_rate']
    for field in required_fields:
        if field not in user_input:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        # Step 2: Call Triage Range API
        triage_range_response = requests.post(TRIAGE_RANGE_API, json=user_input)
        triage_range_response.raise_for_status()  # Raise an exception for HTTP errors
        triage_range_data = triage_range_response.json()

        # Extract information from the Triage Range API response
        triage_range = triage_range_data.get("triage_range")
        confidence_score_range = triage_range_data.get("confidence_score")
        redirect_api = triage_range_data.get("redirect_api")

        # Log the Triage Range API response
        print(f"Triage Range API Response: {triage_range_data}")

        if not redirect_api:
            return jsonify({"error": "Triage Range API did not return a valid redirect_api"}), 500

        # If triage_range is 2, directly set triage_level to 3 and return
        if triage_range == "Range 2":
            final_response = {
                "triage_range": triage_range,
                "confidence_score_range": confidence_score_range,
                "triage_level": 3,  # Direct mapping from range 2 to level 3
                "confidence_score_level": confidence_score_range,  # Assuming confidence is the same
                "explanation": "Triage range indicates direct assignment to level 3."
            }
            return jsonify(final_response)

        # Step 3: Call the Relevant Triage Level API
        triage_level_response = requests.post(redirect_api, json=user_input)
        triage_level_response.raise_for_status()  # Raise an exception for HTTP errors
        triage_level_data = triage_level_response.json()

        # Log the Triage Level API response
        print(f"Triage Level API Response: {triage_level_data}")

        # Step 4: Combine and Return the Final Response
        final_response = {
            "triage_range": triage_range,
            "confidence_score_range": confidence_score_range,
            "triage_level": triage_level_data.get("triage_level"),
            "confidence_score_level": triage_level_data.get("confidence_score"),
            "explanation": triage_level_data.get("explanation")  # Optional: Explanation from the Triage Level API
        }

        return jsonify(final_response)

    except requests.exceptions.RequestException as e:
        print(f"Error while calling API: {e}")
        return jsonify({"error": "Failed to call one of the APIs", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=9000)
