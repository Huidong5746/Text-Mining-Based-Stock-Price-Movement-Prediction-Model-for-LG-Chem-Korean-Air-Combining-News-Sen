from __future__ import annotations

from django.urls import path

from dashboard.views import index


urlpatterns = [
    path("", index, name="dashboard"),
]
