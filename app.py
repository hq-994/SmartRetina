from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

from utils.predict_amd import predict_amd
from utils.predict_dr import predict_dr
from utils.predict_gl import predict_gl

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = ""
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # Predict using all models
                amd_result = predict_amd(file_path)
                dr_result = predict_dr(file_path)
                gl_result = predict_gl(file_path)

                # Extract class and confidence
                gl_label, gl_conf = gl_result.rsplit(" (", 1)
                dr_label, dr_conf = dr_result.rsplit(" (", 1)
                amd_label, amd_conf = amd_result.rsplit(" (", 1)

                gl_label = gl_label.strip()
                dr_label = dr_label.strip()
                amd_label = amd_label.strip()

                gl_conf = float(gl_conf.strip(")%"))
                dr_conf = float(dr_conf.strip(")%"))
                amd_conf = float(amd_conf.strip(")%"))

                filtered = []

                # Apply custom confidence thresholds
                if gl_label != "No Glaucoma" and gl_conf >= 99:
                    filtered.append({"disease": "Glaucoma", "label": gl_label, "confidence": gl_conf})

                if dr_label != "No DR" and dr_conf >= 97:
                    filtered.append({"disease": "Diabetic Retinopathy", "label": dr_label, "confidence": dr_conf})

                if amd_label == "Wet AMD" and amd_conf >= 99:
                    filtered.append({"disease": "AMD", "label": amd_label, "confidence": amd_conf})
                elif amd_label == "Dry AMD" and amd_conf >= 85:
                    filtered.append({"disease": "AMD", "label": amd_label, "confidence": amd_conf})

                if filtered:
                    top = max(filtered, key=lambda x: x["confidence"])
                    if top["disease"] == "Diabetic Retinopathy":
                        prediction = f"Prediction: {top['disease']} ({top['label']})"
                    elif top["disease"] == "AMD":
                        prediction = f"Prediction: {top['label']}"  # e.g., Dry AMD or Wet AMD
                    else:
                        prediction = f"Prediction: {top['disease']}"
                    confidence = f"Confidence: {top['confidence']:.2f}%"
                else:
                    prediction = "Prediction: Healthy"
                    confidence = f"Confidence: {(gl_conf + dr_conf + amd_conf) / 3:.2f}%"

                image_path = file_path

            except Exception as e:
                prediction = f"❌ Error: {str(e)}"
                confidence = ""

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
