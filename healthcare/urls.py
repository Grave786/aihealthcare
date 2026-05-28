from django.urls import path

from . import views


urlpatterns = [
    path("", views.home_view, name="home"),
    path("register", views.register_view, name="register"),
    path("login", views.login_view, name="login"),
    path("logout", views.logout_view, name="logout"),
    path("predict", views.predict_view, name="predict"),
    path("result", views.result_view, name="result"),
    path("history", views.history_view, name="history"),
]
