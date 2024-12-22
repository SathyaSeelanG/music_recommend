pip install -r requirements.txt
pip install moviepy==1.0.5
pip install flask fer[mtcnn] spotipy youtube-search opencv-python-headless tensorflow    

pip install moviepy==1.0.5
python -m pip install --upgrade pip
python app.py
http://127.0.0.1:5000/
export SPOTIFY_CLIENT_ID="your_client_id"
export SPOTIFY_CLIENT_SECRET="your_client_secret"
# First time setup:
1. Install requirements:
   pip install -r requirements.txt

2. Train the model (only needed once):
   python train_model.py

3. Run the application:
   python app.py

# Subsequent runs:
Just run: python app.py