from django.contrib.auth.models import User
from django.db import models


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    disease = models.CharField(max_length=50)
    probability = models.FloatField()
    risk_level = models.CharField(max_length=20)
    input_data = models.JSONField()
    recommendation = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} - {self.disease} ({self.probability}%)"
