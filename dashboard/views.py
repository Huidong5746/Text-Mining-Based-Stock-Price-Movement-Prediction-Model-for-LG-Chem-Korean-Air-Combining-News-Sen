from __future__ import annotations

from django.shortcuts import render


def index(request):
    return render(
        request,
        "dashboard/index.html",
        {
            "api_base_url": "http://127.0.0.1:8001",
        },
    )
