import os
import joblib
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.http import HttpResponse
from django.template import loader
import matplotlib.pyplot as plt
import numpy as np
import io
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import CustomUserCreationForm
from django.contrib.auth.decorators import login_required
import urllib, base64
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def base(request):
    template = loader.get_template('base.html')
    return HttpResponse(template.render())

@login_required
def home(request):
    template = loader.get_template('home.html')
    return HttpResponse(template.render())

def signup_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('home')
    else:
        form = CustomUserCreationForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

def test_view(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES.get('image')
        if uploaded_file:
            # Save uploaded file
            file_path = default_storage.save('tmp/' + uploaded_file.name, ContentFile(uploaded_file.read()))
            context['uploaded'] = default_storage.url(file_path)
            
            # Example prediction logic (replace with actual model prediction)
            prediction = "Positive"  # Placeholder for actual prediction
            context['prediction'] = prediction

    return render(request, 'test.html', context)

def handle_uploaded_file(f, path):
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@login_required
def train(request):
    if request.method == 'POST' and request.FILES.getlist('dataset'):
        dataset_files = request.FILES.getlist('dataset')
        
        cancer_path = os.path.join(settings.MEDIA_ROOT, 'dataset/cancer')
        normal_path = os.path.join(settings.MEDIA_ROOT, 'dataset/normal')
        os.makedirs(cancer_path, exist_ok=True)
        os.makedirs(normal_path, exist_ok=True)

        for file in dataset_files:
            if 'cancer' in file.name.lower():
                handle_uploaded_file(file, os.path.join(cancer_path, file.name))
            elif 'normal' in file.name.lower():
                handle_uploaded_file(file, os.path.join(normal_path, file.name))
        # Extract features and labels
        X = []
        y = []
        for category, path in [('cancer', cancer_path), ('normal', normal_path)]:
            label = 1 if category == 'cancer' else 0
            for filename in os.listdir(path):
                img_path = os.path.join(path, filename)
                img = Image.open(img_path).convert('L')
                img = img.resize((150, 150))
                img_array = np.array(img)
                features = hog(img_array, block_norm='L2-Hys', pixels_per_cell=(16, 16))
                X.append(features)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the models
        models = {
            'RandomForest': RandomForestClassifier(),
            'KNN': KNeighborsClassifier(),
            'SVC': SVC(),
            'DecisionTree': DecisionTreeClassifier()
        }

        history = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            history[name] = accuracy
            # Save each model to a file
            model_path = os.path.join(settings.MEDIA_ROOT, f'model_{name}.joblib')
            joblib.dump(model, model_path)

        # Plot the accuracy
        plt.figure(figsize=(10, 6))
        bars = plt.bar(history.keys(), history.values(), color=['blue', 'green', 'red', 'purple'])

        # Add labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

        plt.xlabel('Algorithm')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.ylim(0, 1)

        # Save accuracy plot as base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)

        return render(request, 'train.html', {'data': uri})

    return render(request, 'train.html')

@login_required
def test(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        file_path = default_storage.save('tmp/' + image_file.name, ContentFile(image_file.read()))
        img_path = default_storage.path(file_path)  # Use default_storage.path to get the correct file path

        img = Image.open(img_path).convert('L')
        img = img.resize((150, 150))
        img_array = np.array(img)
        features = hog(img_array, block_norm='L2-Hys', pixels_per_cell=(16, 16))

        # Load the best model (this is an example, you may want to use a saved model)
        model_path = os.path.join(settings.MEDIA_ROOT, 'model_RandomForest.joblib')  # Replace with the actual model you want to load
        model = joblib.load(model_path)

        prediction = model.predict([features])[0]
        prediction_text = "Cancer" if prediction == 1 else "No Cancer"

        return render(request, 'test.html', {'prediction': prediction_text, 'uploaded': True})

    return render(request, 'test.html', {'uploaded': False})
