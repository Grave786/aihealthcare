from django.contrib import admin

from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("user", "disease", "probability", "risk_level", "created_at")
    list_filter = ("disease", "risk_level", "created_at")
    search_fields = ("user__username", "user__email", "disease")
    readonly_fields = ("created_at",)
