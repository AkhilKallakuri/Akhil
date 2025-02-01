# app/urls.py

from django.urls import path
from .views import base, home, train, test, signup_view, login_view, logout_view

urlpatterns = [
    path('', base, name='base'),        # Base or landing page
    path('home/', home, name='home'),   # Home page
    path('train/', train, name='train'),# Train model page
    path('test/', test, name='test'),   # Test image page
    path('signup/', signup_view, name='signup'), # Signup page
    path('login/', login_view, name='login'),     # Login page
    path('logout/', logout_view, name='logout'), # Logout page
]
