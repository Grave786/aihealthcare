from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.shortcuts import get_object_or_404, redirect, render

from .forms import LoginForm, PredictionForm, RegisterForm
from .ml_services import get_recommendation, predict_disease
from .models import Prediction


def home_view(request):
    return render(request, "home.html")


def register_view(request):
    if request.user.is_authenticated:
        return redirect("predict")

    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        name = form.cleaned_data["name"]
        email = form.cleaned_data["email"].lower()
        password = form.cleaned_data["password"]

        try:
            user = User.objects.create_user(
                username=email,
                email=email,
                password=password,
                first_name=name,
            )
        except IntegrityError:
            messages.error(request, "An account with this email already exists.")
        else:
            login(request, user)
            messages.success(request, "Registration successful. You are logged in.")
            return redirect("predict")

    return render(request, "register.html", {"form": form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect("predict")

    form = LoginForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        email = form.cleaned_data["email"].lower()
        password = form.cleaned_data["password"]
        user = authenticate(request, username=email, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, "Logged in successfully.")
            return redirect("predict")
        messages.error(request, "Invalid email or password.")

    return render(request, "login.html", {"form": form})


def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect("home")


@login_required
def predict_view(request):
    form = PredictionForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        try:
            prediction_result = predict_disease(form.cleaned_data)
        except FileNotFoundError as exc:
            messages.error(request, str(exc))
        except Exception as exc:
            messages.error(request, f"Prediction failed: {exc}")
        else:
            input_data = {}
            for key, value in form.cleaned_data.items():
                if value in (None, ""):
                    continue
                input_data[key] = float(value) if not isinstance(value, str) else value

            input_data["diabetes_probability"] = prediction_result[
                "diabetes_probability"
            ]
            input_data["heart_probability"] = prediction_result["heart_probability"]

            recommendation = get_recommendation(
                prediction_result["disease"],
                prediction_result["probability"],
                prediction_result["risk"],
                input_data,
            )

            prediction = Prediction.objects.create(
                user=request.user,
                disease=prediction_result["disease"],
                probability=prediction_result["probability"],
                risk_level=prediction_result["risk"],
                input_data=input_data,
                recommendation=recommendation,
            )
            request.session["latest_prediction_id"] = prediction.id
            return redirect("result")

    return render(
        request,
        "predict.html",
        {
            "form": form,
            "diabetes_fields": PredictionForm.DIABETES_FIELDS,
            "heart_fields": PredictionForm.HEART_FIELDS,
        },
    )


@login_required
def result_view(request):
    prediction_id = request.GET.get("prediction_id") or request.session.get(
        "latest_prediction_id"
    )
    if not prediction_id:
        messages.info(request, "Please complete a prediction first.")
        return redirect("predict")

    prediction = get_object_or_404(
        Prediction,
        id=prediction_id,
        user=request.user,
    )
    return render(request, "result.html", {"prediction": prediction})


@login_required
def history_view(request):
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, "history.html", {"predictions": predictions})
