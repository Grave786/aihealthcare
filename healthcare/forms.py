from django import forms


class RegisterForm(forms.Form):
    name = forms.CharField(max_length=150)
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)


class LoginForm(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)


class PredictionForm(forms.Form):
    PREDICTION_CHOICES = [
        ("diabetes", "Diabetes"),
        ("heart", "Heart Disease"),
    ]

    DIABETES_FIELDS = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree_function",
        "diabetes_age",
    ]

    HEART_FIELDS = [
        "heart_age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    prediction_type = forms.ChoiceField(
        choices=PREDICTION_CHOICES,
        widget=forms.RadioSelect,
        label="Prediction Type",
    )

    pregnancies = forms.IntegerField(
        min_value=0,
        label="Pregnancies",
        required=False,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "placeholder": "Number of times pregnant (0 if none)",
            }
        ),
    )
    glucose = forms.FloatField(
        min_value=0,
        label="Glucose",
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Blood sugar level (e.g. 120)"}
        ),
    )
    blood_pressure = forms.FloatField(
        min_value=0,
        label="Blood Pressure",
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Blood pressure (e.g. 70)"}
        ),
    )
    skin_thickness = forms.FloatField(
        min_value=0,
        label="Skin Thickness",
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Skin thickness (optional)"}
        ),
    )
    insulin = forms.FloatField(
        min_value=0,
        label="Insulin",
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Insulin level (optional)"}
        ),
    )
    bmi = forms.FloatField(
        min_value=0,
        label="BMI",
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Body Mass Index (e.g. 25.5)"}
        ),
    )
    diabetes_pedigree_function = forms.FloatField(
        min_value=0,
        label="Family History Score",
        required=False,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control",
                "placeholder": "Family diabetes history score",
            }
        ),
    )
    diabetes_age = forms.IntegerField(
        min_value=0,
        label="Age",
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "Your age"}),
    )

    heart_age = forms.IntegerField(
        min_value=0,
        label="Age",
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "Your age"}),
    )
    sex = forms.ChoiceField(
        choices=[("", "Select sex"), ("1", "Male"), ("0", "Female")],
        label="Sex",
        required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    cp = forms.ChoiceField(
        choices=[
            ("", "Select chest pain type"),
            ("0", "Typical angina"),
            ("1", "Atypical angina"),
            ("2", "Non-anginal pain"),
            ("3", "Asymptomatic"),
        ],
        label="Chest Pain Type",
        required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    trestbps = forms.FloatField(
        min_value=0,
        label="Resting Blood Pressure",
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Resting blood pressure"}
        ),
    )
    chol = forms.FloatField(
        min_value=0,
        label="Cholesterol",
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Cholesterol level"}
        ),
    )
    fbs = forms.ChoiceField(
        choices=[("", "High blood sugar?"), ("1", "Yes"), ("0", "No")],
        label="High Blood Sugar",
        required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    restecg = forms.ChoiceField(
        choices=[
            ("", "Select ECG result"),
            ("0", "Normal"),
            ("1", "ST-T wave abnormality"),
            ("2", "Left ventricular hypertrophy"),
        ],
        label="Resting ECG Results",
        required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    thalach = forms.FloatField(
        min_value=0,
        label="Maximum Heart Rate",
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Max heart rate achieved"}
        ),
    )
    exang = forms.ChoiceField(
        choices=[("", "Exercise pain?"), ("1", "Yes"), ("0", "No")],
        label="Exercise Pain",
        required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    oldpeak = forms.FloatField(
        min_value=0,
        label="ST Depression",
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "ST depression (e.g. 1.0)"}
        ),
    )
    slope = forms.ChoiceField(
        choices=[
            ("", "Select slope"),
            ("0", "Upsloping"),
            ("1", "Flat"),
            ("2", "Downsloping"),
        ],
        label="Slope",
        required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    ca = forms.IntegerField(
        min_value=0,
        max_value=4,
        label="Major Vessels",
        required=False,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Number of vessels (0-4)"}
        ),
    )
    thal = forms.ChoiceField(
        choices=[
            ("", "Select thal result"),
            ("0", "Unknown"),
            ("1", "Normal"),
            ("2", "Fixed defect"),
            ("3", "Reversible defect"),
        ],
        label="Thal",
        required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    def clean(self):
        cleaned_data = super().clean()
        prediction_type = cleaned_data.get("prediction_type")

        required_fields = []
        if prediction_type in ("diabetes", "both"):
            required_fields.extend(self.DIABETES_FIELDS)
        if prediction_type in ("heart", "both"):
            required_fields.extend(self.HEART_FIELDS)

        for field_name in required_fields:
            if cleaned_data.get(field_name) in (None, ""):
                self.add_error(field_name, "This field is required for this prediction.")

        return cleaned_data
