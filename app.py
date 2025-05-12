from flask import Flask, render_template, request, jsonify
from models.predictor import CropProductionPredictor
import os

app = Flask(__name__)

# Initialize the predictor
MODEL_FILE = 'data/crop_production_model.pkl'
DATA_PATH = 'data/crop_production.csv'

predictor = CropProductionPredictor()

if not os.path.exists(MODEL_FILE):
    predictor.load_data(DATA_PATH)
    predictor.train_model()
    predictor.save_model(MODEL_FILE)
else:
    predictor.load_model(MODEL_FILE)
    if os.path.exists(DATA_PATH):
        predictor.load_data(DATA_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        state_name = request.form['state']
        district_name = request.form['district']
        season = request.form['season']
        crop = request.form['crop']
        area = float(request.form['area'])
        
        if area <= 0:
            return jsonify({'error': 'Area must be a positive number.'})
        
        production = predictor.predict_production(
            state_name=state_name,
            district_name=district_name,
            season=season,
            crop=crop,
            area=area
        )
        
        return jsonify({'success': True, 'production': round(production, 2)})
    
    except ValueError:
        return jsonify({'error': 'Please enter a valid number for Area.'})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

@app.route('/get_districts', methods=['POST'])
def get_districts():
    state = request.form['state']
    districts = predictor.get_districts_for_state(state)
    return jsonify({'districts': districts})

@app.route('/get_data', methods=['GET'])
def get_data():
    return jsonify({
        'states': predictor.get_states(),
        'crops': predictor.get_crops(),
        'seasons': predictor.get_seasons()
    })

if __name__ == '__main__':
    app.run(debug=True)